"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import torch
from torch import nn
import torch.utils.data
from collections import OrderedDict, defaultdict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize, get_world_size
from pointcept.utils.cache import shared_dict
from pointcept.utils.misc import is_parallel_model, is_pytorch_model, unwrap_model

import pointcept.utils.comm as comm
from pointcept.engines.test import TESTERS

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        # MSC pretrain do not have offset information. Comment the code for support MSC
        # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
        #        "Scan {batch_size} ({points_num}) ".format(
        #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
        #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
        #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
        #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
        # )
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class OpenSegCheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if self.trainer.cfg.eval_only:
            return
        if is_main_process():
            is_best = defaultdict(lambda: False)
            if self.trainer.cfg.evaluate:
                for m_i, (current_metric_value, current_metric_name) in enumerate(
                    zip(
                        self.trainer.comm_info["current_metric_value"],
                        self.trainer.comm_info["current_metric_name"],
                    )
                ):
                    if current_metric_value > self.trainer.best_metric_value[m_i]:
                        self.trainer.best_metric_value[m_i] = current_metric_value
                        is_best[m_i] = True
                        all_metrics = ", ".join(
                            f"{name}: {value:.4f}"
                            for name, value in zip(
                                self.trainer.comm_info["current_metric_name"],
                                self.trainer.comm_info["current_metric_value"],
                            )
                        )
                        self.trainer.other_metric_snapshot[current_metric_name] = (
                            all_metrics
                        )
                        self.trainer.logger.info(
                            "Best validation {} updated to: {:.4f}, All Metrics: {}".format(
                                current_metric_name, current_metric_value, all_metrics
                            )
                        )
                    self.trainer.logger.info(
                        "Currently Best {}: {:.4f}, At That Time: {}".format(
                            current_metric_name,
                            self.trainer.best_metric_value[m_i],
                            self.trainer.other_metric_snapshot.get(
                                current_metric_name, ""
                            ),
                        )
                    )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            save_dict = {
                "epoch": self.trainer.epoch + 1,
                "state_dict": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "scaler": (
                    self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "best_metric_value": self.trainer.best_metric_value,
                "other_metric_snapshot": self.trainer.other_metric_snapshot,
            }
            if is_pytorch_model(self.trainer.recognizer):
                save_dict.update(
                    {"recognizer_state": self.trainer.recognizer.state_dict()}
                )
                trainer_recognizer = unwrap_model(self.trainer.recognizer)
                if hasattr(trainer_recognizer, "class_means"):
                    save_dict.update(
                        {"recognizer_class_means": trainer_recognizer.class_means}
                    )
                if hasattr(trainer_recognizer, "class_covs"):
                    save_dict.update(
                        {"recognizer_class_covs": trainer_recognizer.class_covs}
                    )
            torch.save(save_dict, filename + ".tmp")
            os.replace(filename + ".tmp", filename)

            for m_i, is_best_i in is_best.items():
                if is_best_i:
                    current_metric_name = self.trainer.comm_info["current_metric_name"][
                        m_i
                    ]
                    self.trainer.logger.info(
                        f"Saving model for {current_metric_name} at epoch {self.trainer.epoch+1}."
                    )
                    model_name = f"model_best_{current_metric_name}.pth"
                    shutil.copyfile(
                        filename,
                        os.path.join(self.trainer.cfg.save_path, "model", model_name),
                    )
                    if current_metric_name in [
                        "aupr",
                        "auroc",
                    ] and self.trainer.epoch > int(self.trainer.max_epoch * 0.55):
                        shutil.copyfile(
                            filename,
                            os.path.join(
                                self.trainer.cfg.save_path,
                                "model",
                                f"model_best_{current_metric_name}_ep{self.trainer.epoch+1}.pth",
                            ),
                        )

            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class IncrSegCheckpointSaver(HookBase):
    def __init__(
        self, save_freq=None, tracked_best_metrics=[], tracked_epoch=float("inf")
    ):
        self.save_freq = save_freq
        self.tracked_best_metrics = tracked_best_metrics
        self.tracked_epoch = tracked_epoch

    def before_train(self):
        self.evaluate = self.trainer.cfg.evaluate
        self.evaluate_only = self.trainer.cfg.eval_only
        self.logger = self.trainer.logger
        self.save_path = self.trainer.cfg.save_path

    def after_epoch(self):
        if self.evaluate_only:
            return
        if is_main_process():
            flags_best = self.update_best_metric(
                self.trainer.comm_info["current_metric_value"],
                self.trainer.comm_info["current_metric_name"],
                self.trainer.best_metric_value,
                self.trainer.other_metric_snapshot,
            )
            save_dict = {
                "epoch": self.trainer.epoch + 1,
                "state_dict": self.trainer.incr_learner.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "scaler": (
                    self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "best_metric_value": self.trainer.best_metric_value,
                "other_metric_snapshot": self.trainer.other_metric_snapshot,
            }
            self.save_checkpoint(self.trainer.epoch + 1, save_dict)
            self.save_best_checkpoint(
                flags_best,
                self.trainer.comm_info["current_metric_name"],
                self.trainer.epoch + 1,
            )

    def update_best_metric(self, metric_value, metric_name, best_metric, snapshot={}):
        flags_best = defaultdict(lambda: False)
        if self.evaluate:
            for m_i, (cur_metric_value, cur_metric_name) in enumerate(
                zip(metric_value, metric_name)
            ):
                if cur_metric_value > best_metric[m_i]:
                    best_metric[m_i] = cur_metric_value
                    flags_best[m_i] = True
                    all_metrics = ", ".join(
                        f"{name}: {value:.4f}"
                        for name, value in zip(metric_name, metric_value)
                    )
                    snapshot[cur_metric_name] = all_metrics
                    self.logger.info(
                        "Best validation {} updated to: {:.4f}, All Metrics: {}".format(
                            cur_metric_name, cur_metric_value, all_metrics
                        )
                    )
                self.logger.info(
                    "Currently Best: {}: {:.4f}, At That Time: {}".format(
                        cur_metric_name,
                        best_metric[m_i],
                        snapshot.get(cur_metric_name, ""),
                    )
                )
        return flags_best

    def save_checkpoint(self, epoch, save_dict):
        filename = os.path.join(self.save_path, "model", "model_last.pth")
        self.logger.info("Saving checkpoint to: " + filename)
        torch.save(save_dict, filename + ".tmp")
        os.replace(filename + ".tmp", filename)
        if self.save_freq and epoch % self.save_freq == 0:
            shutil.copyfile(
                filename,
                os.path.join(
                    self.save_path,
                    "model",
                    f"epoch_{epoch}.pth",
                ),
            )

    def save_best_checkpoint(self, flags_best, metric_name, epoch):
        filename = os.path.join(self.save_path, "model", "model_last.pth")
        for m_i, flag_i in flags_best.items():
            if flag_i:
                cur_metric_name = metric_name[m_i]
                self.logger.info(
                    f"Saving model for {cur_metric_name} at epoch {epoch}."
                )
                model_name = f"model_best_{cur_metric_name}.pth"
                shutil.copyfile(
                    filename,
                    os.path.join(self.save_path, "model", model_name),
                )
                if (
                    cur_metric_name in self.tracked_best_metrics
                    and epoch > self.tracked_epoch
                ):
                    shutil.copyfile(
                        filename,
                        os.path.join(
                            self.save_path,
                            "model",
                            f"model_best_{cur_metric_name}_ep{epoch}.pth",
                        ),
                    )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class OpenSegCheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )

            weight = self.replace_key(
                checkpoint["state_dict"], self.keywords, self.replacement
            )
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")

            if checkpoint.get("recognizer_state", None) is not None:
                recognizer_weight = self.replace_key(
                    checkpoint["recognizer_state"], self.keywords, self.replacement
                )
                load_state_info = self.trainer.recognizer.load_state_dict(
                    recognizer_weight, strict=self.strict
                )
                self.trainer.logger.info(
                    f"Missing recognizer keys: {load_state_info[0]}"
                )
                trainer_recognizer = unwrap_model(self.trainer.recognizer)
                if checkpoint.get("recognizer_class_means", None) is not None:
                    trainer_recognizer.class_means = checkpoint[
                        "recognizer_class_means"
                    ]
                if checkpoint.get("recognizer_class_covs", None) is not None:
                    trainer_recognizer.class_covs = checkpoint["recognizer_class_covs"]

            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                best_value = checkpoint["best_metric_value"]
                if isinstance(best_value, dict):
                    self.trainer.best_metric_value.update(best_value)
                else:
                    # Compatible with old versions
                    self.trainer.best_metric_value[0] = best_value
                self.trainer.other_metric_snapshot.update(
                    checkpoint.get("other_metric_snapshot", {})
                )
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")

    def replace_key(self, state_dict, keywords, replacement):
        weight = OrderedDict()
        for key, value in state_dict.items():
            if not key.startswith("module."):
                # if comm.get_world_size() > 1:
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if keywords in key:
                key = key.replace(keywords, replacement)
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
            weight[key] = value
        return weight


@HOOKS.register_module()
class IncrSegCheckpointLoader(OpenSegCheckpointLoader):
    def __init__(self, keywords="", replacement=None, strict=False):
        super().__init__()

    def before_train(self):
        self.logger = self.trainer.logger
        cfg = self.trainer.cfg

        self.logger.info(
            "=> Loading base model and incremental learner checkpoint & weight ..."
        )

        if cfg.incr_resume and cfg.resume:
            raise RuntimeError(
                "Incremental model can not resume from base model weight and incremental learner weight at the same time."
            )

        base_ckpt = self._load_ckpt_if_exist(cfg.base_ckpt, "base")

        if cfg.incr_resume:  # TODO debug
            incr_ckpt = self._load_ckpt_if_exist(cfg.incr_ckpt, "incremental learner")
            self.load_incremental_weight(
                incr_ckpt["state_dict"], base_ckpt.get("state_dict")
            )
            self.logger.info(
                f"Resuming training based on incremental learner checkpoint from {incr_ckpt['epoch']} epoch"
            )
            self.resume_training(incr_ckpt)
        elif cfg.load_base_weight_to_incr_learner:
            self.logger.info("Loading base model weight to incremental learner ...")
            assert base_ckpt, "Base model weight is required for incremental model."
            incr_weight = getattr(self, cfg.base_weight_process_func)(
                base_ckpt["state_dict"]
            )
            self.load_incremental_weight(incr_weight, base_ckpt["state_dict"])
            if cfg.resume:
                self.logger.info(
                    f"Resume training based on base model from {base_ckpt['epoch']} epoch"
                )
                self.resume_training(base_ckpt)
        else:
            self.logger.info("Incremental model weight is not provided.")
            self.logger.info("Loading base checkpoint to trainer model ...")
            if not is_parallel_model(self.trainer.model):
                base_weight = self.replace_key(base_ckpt["state_dict"], "module.", "")
            load_state_info = self.trainer.model.load_state_dict(
                base_weight, strict=self.strict
            )
            self.logger.info(f"Missing keys: {load_state_info[0]}")

    def _load_ckpt_if_exist(self, ckpt_path, name):
        if ckpt_path:
            if os.path.isfile(ckpt_path):
                self.logger.info(f"Loading {name} checkpoint at: {ckpt_path}")
                return torch.load(
                    ckpt_path, map_location=lambda storage, loc: storage.cuda()
                )
            else:
                raise RuntimeError(f"No {name} checkpoint found at: {ckpt_path}")
        return {}

    def load_weight(self, weight, model):
        self.logger.info(
            f"Loading weights with keyword: {self.keywords}, replace keyword with: {self.replacement}"
        )
        weight = self.replace_key(weight, self.keywords, self.replace_key)
        load_state_info = model.load_state_dict(weight, strict=self.strict)
        self.logger.info(f"Missing keys: {load_state_info[0]}")

    def load_incremental_weight(self, incr_weight, base_weight={}):
        self.logger.info(
            f"Loading incremental learner weight ..., keyword replacement: {self.keywords} -> {self.replacement}"
        )
        if unwrap_model(self.trainer.incr_learner).need_teacher_model:
            teacher_weight = self.replace_key(
                base_weight, "backbone", "teacher_model.backbone"
            )
            merge_weight = self.replace_key(
                {**incr_weight, **teacher_weight}, self.keywords, self.replacement
            )
        load_state_info = self.trainer.incr_learner.load_state_dict(
            merge_weight, strict=self.strict
        )
        self.logger.info(f"Missing keys: {load_state_info[0]}")

    def trim_base_weight_head(self, weight):
        self.logger.info(f"Process base weights by trimming the head layer ...")
        weight = self.replace_key(weight, "backbone", "incr_backbone")
        new_state_dict = {}
        model_state_dict = self.trainer.incr_learner.state_dict()
        for k, v in weight.items():
            if k not in model_state_dict:
                self.logger.warning(f"[Skip] '{k}' not in model.")
                continue

            model_param = model_state_dict[k]
            if v.shape == model_param.shape:
                new_state_dict[k] = v
                self.logger.debug(f"[Keep] {k}: {v.shape}")
            elif (
                v.ndim == model_param.ndim
                and v.shape[1:] == model_param.shape[1:]
                and v.shape[0] <= model_param.shape[0]
            ):
                new_param = model_param.clone()
                new_param[: v.shape[0]] = v
                new_state_dict[k] = new_param
                self.logger.info(
                    f"[Partial load] '{k}': base {v.shape} -> new {model_param.shape}, copied {v.shape[0]} rows"
                )
            else:
                self.logger.debug(
                    f"[Shape Mismatch] {k}: {v.shape} vs {model_param.shape} (Skipped)"
                )
        return new_state_dict

    def reserve_matched(self, weight):  # TODO debug
        self.logger.info(f"Process base weights by reserving matched weights ...")
        weight = self.replace_key(weight, "backbone", "incr_backbone")
        new_state_dict = {}
        model_state_dict = self.trainer.incr_learner.state_dict()
        for k, v in weight.items():
            if k in model_state_dict:
                model_param = model_state_dict[k]
                if v.shape == model_param.shape:
                    new_state_dict[k] = v
                    self.logger.debug(f"[Keep] {k}: {v.shape}")
                else:
                    self.logger.debug(
                        f"[Shape Mismatch] {k}: {v.shape} vs {model_param.shape} (Skipped)"
                    )
            else:
                self.logger.debug(f"[Missing] {k} not in model (Skipped)")
        return weight

    def resume_training(self, checkpoint):
        self.trainer.start_epoch = checkpoint["epoch"]
        best_value = checkpoint["best_metric_value"]
        if isinstance(best_value, dict):
            self.trainer.best_metric_value.update(best_value)
        else:
            # Compatible with old versions
            self.trainer.best_metric_value[0] = best_value
        self.trainer.other_metric_snapshot.update(
            checkpoint.get("other_metric_snapshot", {})
        )
        self.safe_load_optimizer_state(self.trainer.optimizer, checkpoint["optimizer"])
        self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.trainer.cfg.enable_amp:
            self.trainer.scaler.load_state_dict(checkpoint["scaler"])

    def safe_load_optimizer_state(self, optimizer, loaded_state):
        try:
            optimizer.load_state_dict(loaded_state)
        except ValueError as e:
            self.logger.warning(
                "Optimizer state_dict load failed, attempting partial load."
            )
            # Try the recovery of public parameters in param_groups
            new_param_groups = optimizer.state_dict()["param_groups"]
            for new_group, loaded_group in zip(
                new_param_groups, loaded_state["param_groups"]
            ):
                for key in loaded_group:
                    if (
                        key != "params"
                    ):  # params key must be consistent with current model
                        new_group[key] = loaded_group[key]
            # update optimizer state_dict
            optimizer.load_state_dict({"state": {}, "param_groups": new_param_groups})
            self.logger.info("Optimizer state partial load completed.")
        except Exception as e:
            self.logger.error(
                f"Failed to load optimizer state_dict: {e}. No recovery attempted."
            )


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path)
            state_dict = checkpoint["state_dict"]
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "").split(".")[0]
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            for data_path in self.data_list:
                cache_name = self.get_cache_name(data_path)
                data = torch.load(data_path)
                shared_dict(cache_name, data)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)
