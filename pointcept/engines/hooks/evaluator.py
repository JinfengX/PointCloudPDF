"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import (
    aupr_and_auroc,
    intersection_and_union_gpu,
    is_pytorch_model,
    selected_mask,
)

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class OpenSegEvaluator(HookBase):
    def before_train(self):
        self.num_classes = self.trainer.cfg.data.num_classes
        self.ignore_index = self.trainer.cfg.data.ignore_index
        self.unknown_label = self.trainer.cfg.unknown_label
        self.mask_known = ~selected_mask(self.unknown_label, self.num_classes)

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        # torch.cuda.empty_cache()
        self.trainer.model.eval()
        if is_pytorch_model(self.trainer.recognizer):
            self.trainer.recognizer.eval()
        all_score = []
        all_segment = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            self.trainer.input_to_device(input_dict)
            self.trainer.label_rename(input_dict)
            with torch.no_grad():
                model_output = self.trainer.model(input_dict)
                recognizer_output = self.trainer.recognizer(input_dict)
            output = model_output["seg_logits"]
            loss = model_output["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment_oracle"]
            score = recognizer_output["score"]
            all_score.append(score.clone().detach().cpu())
            all_segment.append(segment.clone().detach().cpu())
            if "origin_coord" in input_dict.keys():
                raise NotImplementedError("Not used yet")

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.segmentation_metric(pred, segment, self.num_classes, self.ignore_index)
            self.recognition_metric(score, segment)

            if "origin_coord" in input_dict.keys():
                raise NotImplementedError("Not used yet")

            self.trainer.logger.info(
                f"Test: [{i + 1}/{len(self.trainer.val_loader)}] Loss {loss.item():.4f}"
            )

        loss_avg = self.trainer.storage.history("val_loss").avg

        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class[self.mask_known])
        m_acc = np.mean(acc_class[self.mask_known])
        all_acc = sum(intersection[self.mask_known]) / (
            sum(target[self.mask_known]) + 1e-10
        )

        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        self.trainer.logger.info(
            "Val result: aupr/auroc {:.4f}/{:.4f}".format(
                self.trainer.storage.history("val_aupr").avg,
                self.trainer.storage.history("val_auroc").avg,
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )

        # all_score = comm.gather(all_score, dst=0)
        # all_segment = comm.gather(all_segment, dst=0)
        # if comm.is_main_process():
        #     all_score = torch.cat([torch.cat(item, dim=0) for item in all_score], dim=0)
        #     all_segment = torch.cat(
        #         [torch.cat(item, dim=0) for item in all_segment], dim=0
        #     )
        #     all_aupr, all_auroc = aupr_and_auroc(
        #         all_score, all_segment, self.unknown_label, self.ignore_index
        #     )
        #     self.trainer.logger.info(
        #         "Val result: all scenes aupr/auroc {:.4f}/{:.4f}".format(
        #             all_aupr,
        #             all_auroc,
        #         )
        #     )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            self.trainer.writer.add_scalar(
                "val/aupr", self.trainer.storage.history("val_aupr").avg, current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/auroc",
                self.trainer.storage.history("val_auroc").avg,
                current_epoch,
            )
            # self.trainer.writer.add_scalar("val/aupr_allScene", all_aupr, current_epoch)
            # self.trainer.writer.add_scalar(
            #     "val/auroc_allScene", all_auroc, current_epoch
            # )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        if comm.is_main_process():
            self.trainer.comm_info["current_metric_value"] = [
                m_iou,
                self.trainer.storage.history("val_aupr").avg,
                self.trainer.storage.history("val_auroc").avg,
            ]  # save for saver
            self.trainer.comm_info["current_metric_name"] = [
                "mIoU",
                "aupr",
                "auroc",
            ]  # save for saver

    def after_train(self):
        if comm.is_main_process():
            for m_i, metric_name in enumerate(
                self.trainer.comm_info["current_metric_name"]
            ):
                self.trainer.logger.info(
                    "Best {}: {:.4f}".format(
                        metric_name, self.trainer.best_metric_value[m_i]
                    )
                )

    def segmentation_metric(self, pred, segment, num_classes, ignore_index):
        intersection, union, target = intersection_and_union_gpu(
            pred,
            segment,
            num_classes,
            ignore_index,
        )
        if comm.get_world_size() > 1:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        self.trainer.storage.put_scalar("val_intersection", intersection)
        self.trainer.storage.put_scalar("val_union", union)
        self.trainer.storage.put_scalar("val_target", target)

        # mIoU = np.mean(intersection[self.mask_known] / (union[self.mask_known] + 1e-10))
        # mAcc = np.mean(
        #     intersection[self.mask_known] / (target[self.mask_known] + 1e-10)
        # )
        # accuracy = sum(intersection[self.mask_known]) / (
        #     sum(target[self.mask_known]) + 1e-10
        # )
        # info = "mIoU: {mIoU:.4f} mAcc: {mAcc:.4f} accuracy: {accuracy:.4f} ".format(
        #     mIoU=mIoU, mAcc=mAcc, accuracy=accuracy
        # )

    def recognition_metric(self, score, segment):
        aupr, auroc = aupr_and_auroc(
            score,
            segment,
            self.unknown_label,
            self.ignore_index,
        )
        aupr_auroc = {"device": comm.get_rank(), "aupr": aupr, "auroc": auroc}
        aupr_auroc = comm.all_gather(aupr_auroc)
        # fmt_aupr, fmt_auroc = [], []
        for item in aupr_auroc:
            aupr_i, auroc_i = item["aupr"], item["auroc"]
            if aupr_i is None and auroc_i is None:
                # fmt_aupr.append("None")
                # fmt_auroc.append("None")
                self.trainer.logger.debug(
                    f"DEVICE: {item['device']} This batch contains no points of unknown classes."
                )
            else:
                # fmt_aupr.append(format(aupr, ".4f"))
                # fmt_auroc.append(format(auroc, ".4f"))
                self.trainer.storage.put_scalar("val_aupr", aupr_i)
                self.trainer.storage.put_scalar("val_auroc", auroc_i)

        # if fmt_aupr != ["None"] * comm.get_world_size():
        #     info = "aupr: {} ({:.4f}) auroc: {} ({:.4f}) ".format(
        #         fmt_aupr,
        #         self.trainer.storage.history("val_aupr").avg,
        #         fmt_auroc,
        #         self.trainer.storage.history("val_auroc").avg,
        #     )


@HOOKS.register_module()
class IncrSegEvaluator(OpenSegEvaluator):
    def __init__(self):
        super().__init__()

    def before_train(self):
        self.base_num_classes = self.trainer.cfg.data.num_classes
        self.remap_num_classes = self.base_num_classes + len(
            self.trainer.cfg.incr_label_remap
        )
        self.ignore_index = self.trainer.cfg.data.ignore_index
        self.unknown_label = self.trainer.cfg.unknown_label
        self.mask_known = ~selected_mask(self.unknown_label, self.base_num_classes)
        self.incr_label_idx = list(self.trainer.cfg.incr_label_remap.values())
        self.mask_incr_remap = ~selected_mask(
            self.unknown_label, self.remap_num_classes
        )
        self.map_reverse = {v: k for k, v in self.trainer.cfg.incr_label_remap.items()}

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        if is_pytorch_model(self.trainer.incr_learner):
            self.trainer.incr_learner.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            self.trainer.input_to_device(input_dict)
            with torch.no_grad():
                incr_output = self.trainer.incr_learner(input_dict)
            incr_seg_logits = incr_output["seg_logits"]
            incr_loss = incr_output["loss"]
            incr_pred = incr_seg_logits.max(1)[1]
            segment = input_dict["segment_incr_remap"]
            if "origin_coord" in input_dict.keys():
                raise NotImplementedError("Not used yet")

            self.trainer.storage.put_scalar("val_loss", incr_loss.item())
            self.segmentation_metric(
                incr_pred, segment, self.remap_num_classes, self.ignore_index
            )

            if "origin_coord" in input_dict.keys():
                raise NotImplementedError("Not used yet")

            self.trainer.logger.info(
                f"Test: [{i + 1}/{len(self.trainer.val_loader)}] Loss {incr_loss.item():.4f}"
            )

        loss_avg = self.trainer.storage.history("val_loss").avg

        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class, acc_class, metric_known, metric_incr, metric_remap = (
            self.incr_segmentation_metric(intersection, union, target)
        )

        # log metrics
        self.trainer.logger.info(
            f"Val result: mIoU/mAcc/Acc known {metric_known['mIoU']:.4f}/{metric_known['mAcc']:.4f}/{metric_known['Acc']:.4f}."
        )
        self.trainer.logger.info(
            f"Val result: mIoU/mAcc/Acc incr {metric_incr['mIoU']:.4f}/{metric_incr['mAcc']:.4f}/{metric_incr['Acc']:.4f}."
        )
        self.trainer.logger.info(
            f"Val result: mIoU/mAcc/Acc remap {metric_remap['mIoU']:.4f}/{metric_remap['mAcc']:.4f}/{metric_remap['Acc']:.4f}."
        )
        for cls_i, (cls_iou, cls_acc) in enumerate(zip(iou_class, acc_class)):
            class_name = self.trainer.cfg.data.names[
                self.map_reverse[cls_i] if cls_i >= self.base_num_classes else cls_i
            ]
            prefix = "Increment " if cls_i >= self.base_num_classes else ""

            self.trainer.logger.info(
                f"{prefix}Class_{cls_i}-{class_name} Result: iou/accuracy {cls_iou:.4f}/{cls_acc:.4f}"
            )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(
                "val/mIoU", metric_known["mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mAcc", metric_known["mAcc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/allAcc", metric_known["Acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mIoU_incr", metric_incr["mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mAcc_incr", metric_incr["mAcc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/allAcc_incr", metric_incr["Acc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mIoU_remap", metric_remap["mIoU"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/mAcc_remap", metric_remap["mAcc"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/allAcc_remap", metric_remap["Acc"], current_epoch
            )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = [
            metric_known["mIoU"],
            metric_incr["mIoU"],
            metric_remap["mIoU"],
        ]  # save for saver
        self.trainer.comm_info["current_metric_name"] = [
            "mIoU_known",
            "mIoU_incr",
            "mIoU_remap",
        ]

    def after_train(self):
        if comm.is_main_process():
            for m_i, metric_name in enumerate(
                self.trainer.comm_info["current_metric_name"]
            ):
                self.trainer.logger.info(
                    "Best {}: {:.4f}".format(
                        metric_name, self.trainer.best_metric_value[m_i]
                    )
                )

    def incr_segmentation_metric(self, intersection, union, target):
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)

        m_iou_known = np.mean(iou_class[: self.base_num_classes][self.mask_known])
        m_acc_known = np.mean(acc_class[: self.base_num_classes][self.mask_known])
        acc_known = sum(intersection[: self.base_num_classes][self.mask_known]) / sum(
            target[: self.base_num_classes][self.mask_known] + 1e-10
        )

        m_iou_incr = np.mean(iou_class[self.incr_label_idx])
        m_acc_incr = np.mean(acc_class[self.incr_label_idx])
        acc_incr = sum(intersection[self.incr_label_idx]) / (
            sum(target[self.incr_label_idx]) + 1e-10
        )

        m_iou_remap = np.mean(iou_class[self.mask_incr_remap])
        m_acc_remap = np.mean(acc_class[self.mask_incr_remap])
        acc_remap = sum(intersection[self.mask_incr_remap]) / (
            sum(target[self.mask_incr_remap]) + 1e-10
        )

        return (
            iou_class,
            acc_class,
            {"mIoU": m_iou_known, "mAcc": m_acc_known, "Acc": acc_known},
            {"mIoU": m_iou_incr, "mAcc": m_acc_incr, "Acc": acc_incr},
            {"mIoU": m_iou_remap, "mAcc": m_acc_remap, "Acc": acc_remap},
        )


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
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
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
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
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
