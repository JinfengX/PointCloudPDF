"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from torch_scatter import scatter_mean

from pointcept.datasets.transform import RemapLabel
from pointcept.incrLearners.builder import build_incremental_learner

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.models.utils import build_model_hook
from pointcept.recognizers import build_recognizer
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    is_pytorch_model,
    make_dirs,
    selected_mask,
    unwrap_model,
    aupr_and_auroc,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class OpenSegTester(TesterBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.model_hooks = self.build_model_hook()
        self.recognizer = self.build_recognizer()
        self.mask_known = ~selected_mask(
            self.cfg.unknown_label, self.cfg.data.num_classes
        )

    def test(self):
        with self.model_hooks:
            self.test_run()

    def test_run(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        aupr_meter = AverageMeter()
        auroc_meter = AverageMeter()
        self.model.eval()
        if is_pytorch_model(self.recognizer):
            self.recognizer.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            score_save_path = os.path.join(save_path, "{}_score.npy".format(data_name))
            if os.path.isfile(pred_save_path) and os.path.isfile(score_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                score = np.load(score_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                score = []
                score_idx = []
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        score_part = self.recognizer(input_dict)["score"]
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be
                    score.append(score_part)
                    score_idx.append(idx_part)

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data.cpu().numpy()
                score = (
                    scatter_mean(
                        torch.cat(score),
                        torch.cat(score_idx),
                        dim=0,
                        dim_size=segment.size,
                    )
                    .cpu()
                    .numpy()
                )
                np.save(pred_save_path, pred)
                np.save(score_save_path, score)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            cls_mask = mask & self.mask_known
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[cls_mask])
            acc = sum(intersection[cls_mask]) / (sum(target[cls_mask]) + 1e-10)

            m_iou = np.mean(
                intersection_meter.sum[cls_mask] / (union_meter.sum[cls_mask] + 1e-10)
            )
            m_acc = np.mean(
                intersection_meter.sum[cls_mask] / (target_meter.sum[cls_mask] + 1e-10)
            )

            aupr, auroc = aupr_and_auroc(
                score, segment, self.cfg.unknown_label, self.cfg.data.ignore_index
            )
            aupr_auroc_dict = {"device": comm.get_rank(), "aupr": aupr, "auroc": auroc}
            aupr_auroc_dict = comm.all_gather(aupr_auroc_dict)
            for aupr_auroc in aupr_auroc_dict:
                aupr_i, auroc_i = aupr_auroc["aupr"], aupr_auroc["auroc"]
                if aupr_i is None and auroc_i is None:
                    logger.debug(
                        f"DEVICE: {aupr_auroc['device']} This batch contains no points of unknown classes"
                    )
                else:
                    aupr_meter.update(aupr_i)
                    auroc_meter.update(auroc_i)
            record[data_name].update(dict(score=score, segment=segment))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f}) "
                "aupr {aupr} ({m_aupr:.4f}) "
                "auroc {auroc} ({m_auroc}:.4f) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                    aupr="{:.4f}".format(aupr) if aupr is not None else "N/A",
                    m_aupr=aupr_meter.avg,
                    auroc="{:.4f}".format(auroc) if aupr is not None else "N/A",
                    m_auroc=auroc_meter.avg,
                )
            )
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
                os.makedirs(
                    os.path.join(
                        save_path,
                        "submit_score",
                        "sequences",
                        sequence_name,
                        "predictions",
                    ),
                    exist_ok=True,
                )
                score.tofile(
                    os.path.join(
                        save_path,
                        "submit_score",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class[self.mask_known])
            mAcc = np.mean(accuracy_class[self.mask_known])
            allAcc = sum(intersection[self.mask_known]) / (
                sum(target[self.mask_known]) + 1e-10
            )

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )

            all_score = np.concatenate(
                [meters["score"] for _, meters in record.items()], axis=0
            )
            all_segment = np.concatenate(
                [meters["segment"] for _, meters in record.items()], axis=0
            )
            all_aupr, all_auroc = aupr_and_auroc(
                all_score, all_segment, self.cfg.unknown_label
            )
            logger.info(
                f"Val result: aupr/auroc {aupr_meter.avg:.4f}/{auroc_meter.avg:.4f}"
            )
            logger.info(
                f"Val result: all points aupr/auroc {all_aupr:.4f}/{all_auroc:.4f}"
            )

            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch

    def build_model_hook(self):
        model_hooks = build_model_hook(self.cfg.model_hooks)
        model_hooks.set_logger(self.logger)
        model_hooks.set_model(self.model)
        self.logger.info(model_hooks)
        return model_hooks

    def build_recognizer(self):
        recognizer = build_recognizer(self.cfg.recognizer)
        recognizer.model_hooks = self.model_hooks
        if isinstance(recognizer, nn.Module):
            n_parameters = sum(
                p.numel() for p in recognizer.parameters() if p.requires_grad
            )
            # logger.info(f"Model: \n{self.model}")
            self.logger.info(f"Num params of recognizer: {n_parameters}")
            recognizer = create_ddp_model(
                recognizer.cuda(),
                broadcast_buffers=False,
                find_unused_parameters=self.cfg.find_unused_parameters,
            )
            if os.path.isfile(self.cfg.weight):
                self.logger.info(f"Loading recognizer weight at: {self.cfg.weight}")
                checkpoint = torch.load(
                    self.cfg.weight, map_location=lambda storage, loc: storage.cuda()
                )
                if checkpoint.get("recognizer_state", None) is not None:
                    weight = OrderedDict()
                    for key, value in checkpoint["recognizer_state"].items():
                        if key.startswith("module."):
                            if comm.get_world_size() == 1:
                                key = key[7:]  # module.xxx.xxx -> xxx.xxx
                        else:
                            if comm.get_world_size() > 1:
                                key = "module." + key  # xxx.xxx -> module.xxx.xxx
                        weight[key] = value
                    load_state_info = recognizer.load_state_dict(weight, strict=True)
                    trainer_recognizer = unwrap_model(recognizer)
                    if checkpoint.get("recognizer_class_means", None) is not None:
                        trainer_recognizer.class_means = checkpoint[
                            "recognizer_class_means"
                        ]
                    if checkpoint.get("recognizer_class_covs", None) is not None:
                        trainer_recognizer.class_covs = checkpoint[
                            "recognizer_class_covs"
                        ]
            else:
                raise RuntimeError(
                    "=> No checkpoint found at '{}'".format(self.cfg.weight)
                )

        return recognizer


@TESTERS.register_module()
class IncrSegTester(TesterBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        from pointcept.engines.hooks.evaluator import IncrSegEvaluator

        class MockTrainer:
            def __init__(self, cfg):
                self.cfg = cfg

        self.incr_learner = self.build_incremental_learner()

        self.trainer = MockTrainer(cfg)
        IncrSegEvaluator.before_train(self)

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.incr_learner.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            data_dict = RemapLabel(self.cfg.incr_label_remap)(data_dict)
            segment = data_dict.pop("segment_incr_remap")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.remap_num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.incr_learner(input_dict)[
                            "seg_logits"
                        ]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.remap_num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou_known = np.mean(
                iou_class[: self.base_num_classes][
                    mask[: self.base_num_classes] & self.mask_known
                ]
            )
            acc_known = sum(
                intersection[: self.base_num_classes][
                    mask[: self.base_num_classes] & self.mask_known
                ]
            ) / (
                sum(
                    target[: self.base_num_classes][
                        mask[: self.base_num_classes] & self.mask_known
                    ]
                )
                + 1e-10
            )
            m_iou_known = np.mean(
                intersection_meter.sum[: self.base_num_classes][self.mask_known]
                / (union_meter.sum[: self.base_num_classes][self.mask_known] + 1e-10)
            )
            m_acc_known = np.mean(
                intersection_meter.sum[: self.base_num_classes][self.mask_known]
                / (target_meter.sum[: self.base_num_classes][self.mask_known] + 1e-10)
            )

            incr_valid = mask[self.base_num_classes :].any()
            incr_mask_indices = np.array(self.incr_label_idx)[
                mask[
                    self.base_num_classes : self.base_num_classes
                    + len(self.incr_label_idx)
                ]
            ]
            iou_incr = np.mean(iou_class[incr_mask_indices]) if incr_valid else -1.0
            acc_incr = (
                sum(intersection[incr_mask_indices])
                / (sum(target[incr_mask_indices]) + 1e-10)
                if incr_valid
                else -1.0
            )
            m_iou_incr = np.mean(
                intersection_meter.sum[self.incr_label_idx]
                / (union_meter.sum[self.incr_label_idx] + 1e-10)
            )
            m_acc_incr = np.mean(
                intersection_meter.sum[self.incr_label_idx]
                / (target_meter.sum[self.incr_label_idx] + 1e-10)
            )

            iou_remap = np.mean(iou_class[mask & self.mask_incr_remap])
            acc_remap = sum(intersection[mask & self.mask_incr_remap]) / (
                sum(target[mask & self.mask_incr_remap]) + 1e-10
            )
            m_iou_remap = np.mean(
                intersection_meter.sum[self.mask_incr_remap]
                / (union_meter.sum[self.mask_incr_remap] + 1e-10)
            )
            m_acc_remap = np.mean(
                intersection_meter.sum[self.mask_incr_remap]
                / (target_meter.sum[self.mask_incr_remap] + 1e-10)
            )

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy known {acc_known:.4f} ({m_acc_known:.4f}) "
                "mIoU known {iou_known:.4f} ({m_iou_known:.4f}) "
                "Accuracy incr {acc_incr:.4f} ({m_acc_incr:.4f}) "
                "mIoU incr {iou_incr:.4f} ({m_iou_incr:.4f}) "
                "mIoU remap {iou_remap:.4f} ({m_iou_remap:.4f}) "
                "Accuracy remap {acc_remap:.4f} ({m_acc_remap:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc_known=acc_known,
                    m_acc_known=m_acc_known,
                    iou_known=iou_known,
                    m_iou_known=m_iou_known,
                    acc_incr=acc_incr,
                    m_acc_incr=m_acc_incr,
                    iou_incr=iou_incr,
                    m_iou_incr=m_iou_incr,
                    acc_remap=acc_remap,
                    m_acc_remap=m_acc_remap,
                    iou_remap=iou_remap,
                    m_iou_remap=m_iou_remap,
                )
            )
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU_known = np.mean(iou_class[: self.base_num_classes][self.mask_known])
            mAcc_known = np.mean(
                accuracy_class[: self.base_num_classes][self.mask_known]
            )
            allAcc_known = sum(
                intersection[: self.base_num_classes][self.mask_known]
            ) / (sum(target[: self.base_num_classes][self.mask_known]) + 1e-10)

            mIoU_incr = np.mean(iou_class[self.incr_label_idx])
            mAcc_incr = np.mean(accuracy_class[self.incr_label_idx])
            allAcc_incr = sum(intersection[self.incr_label_idx]) / (
                sum(target[self.incr_label_idx]) + 1e-10
            )

            mIoU_remap = np.mean(iou_class[self.mask_incr_remap])
            mAcc_remap = np.mean(accuracy_class[self.mask_incr_remap])
            allAcc_remap = sum(intersection[self.mask_incr_remap]) / (
                sum(target[self.mask_incr_remap]) + 1e-10
            )

            logger.info(
                "Val result: mIoU/mAcc/allAcc known {:.4f}/{:.4f}/{:.4f}"
                "Val result: mIoU/mAcc/allAcc incr {:.4f}/{:.4f}/{:.4f}"
                "Val result: mIoU/mAcc/allAcc remap {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU_known,
                    mAcc_known,
                    allAcc_known,
                    mIoU_incr,
                    mAcc_incr,
                    allAcc_incr,
                    mIoU_remap,
                    mAcc_remap,
                    allAcc_remap,
                )
            )
            for i in range(self.cfg.data.num_classes):
                class_name = self.cfg.data.names[
                    self.map_reverse[i] if i >= self.base_num_classes else i
                ]
                prefix = "Increment " if i >= self.base_num_classes else ""
                logger.info(
                    "{prefix}Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        prefix=prefix,
                        idx=i,
                        name=class_name,
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_incremental_learner(self):
        incr_learner = build_incremental_learner(self.cfg.incremental_learner)
        if is_pytorch_model(incr_learner):
            if self.cfg.sync_bn:
                incr_learner = nn.SyncBatchNorm.convert_sync_batchnorm(incr_learner)
            n_parameters = sum(
                p.numel() for p in incr_learner.parameters() if p.requires_grad
            )
            self.logger.info(f"Num params: {n_parameters}")
            incr_learner = create_ddp_model(
                incr_learner.cuda(),
                broadcast_buffers=False,
                find_unused_parameters=self.cfg.find_unused_parameters,
            )
        # load incremental learner weight
        if os.path.isfile(self.cfg.incr_weight):
            self.logger.info(f"Loading weight at: {self.cfg.incr_weight}")
            ckpt = torch.load(
                self.cfg.incr_weight, map_location=lambda storage, loc: storage.cuda()
            )
            weight = OrderedDict()
            for key, value in ckpt["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            incr_learner.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.incr_weight, ckpt["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return incr_learner

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
