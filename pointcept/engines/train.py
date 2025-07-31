"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from collections import defaultdict
from contextlib import ExitStack, nullcontext

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial


if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.models.utils import build_model_hook
from pointcept.recognizers import build_recognizer
from pointcept.incrLearner.builder import build_incremental_learner
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer, build_optimizer_from_named_params
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry
from pointcept.utils.misc import is_pytorch_model, unwrap_model


TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader


@TRAINERS.register_module("OpenSegTrainer")
class OpenSegTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_metric_value = defaultdict(self.default_best_metric_value)
        self.other_metric_snapshot = defaultdict(self.default_other_metric_snapshot)
        self.cfg.eval_only = cfg.get("eval_only", False)
        self.logger.info("=> Building model hooks ...")
        self.model_hooks = self.build_model_hook()
        self.logger.info("=> Building recognizer ...")
        self.recognizer = self.build_recognizer()
        self.optimizer = self.build_open_optimizer()
        # rebuild the scheduler after updating the optimizer.
        self.scheduler = self.build_scheduler()

    def train(self):
        with ExitStack() as stack:
            stack.enter_context(self.model_hooks)
            if self.cfg.eval_only:
                self.storage = stack.enter_context(EventStorage())
                super().before_train()
                self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    super().after_epoch()
            else:
                super().train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model_forward(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def before_epoch(self):
        self.model.train()
        if is_pytorch_model(self.recognizer):
            self.recognizer.train()
        if self.recognizer:
            unwrap_model(self.recognizer).set_epoch(self.epoch)
        super().before_epoch()

    def model_forward(self, input_dict):
        self.label_rename(input_dict)
        model_output = self.model(input_dict)
        recognizer_output = self.recognizer(input_dict)
        if recognizer_output.get("loss", None) is not None:
            model_output["loss"] = model_output["loss"] + recognizer_output["loss"]
            model_output.update({"loss_rec": recognizer_output["loss"]})
        return model_output

    def input_to_device(self, input_dict):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

    def label_rename(self, input_dict):
        # switch known segment to all segment for open-set segmentation
        if "segment_known" in input_dict:
            input_dict.update({"segment_oracle": input_dict["segment"]})
            input_dict["segment"] = input_dict["segment_known"]

    def build_open_optimizer(self):
        all_named_params = dict(self.model.named_parameters())
        if is_pytorch_model(self.recognizer):
            all_named_params.update(dict(self.recognizer.named_parameters()))
        return build_optimizer_from_named_params(
            self.cfg.optimizer, all_named_params, self.cfg.param_dicts
        )

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
            if self.cfg.sync_bn:
                recognizer = nn.SyncBatchNorm.convert_sync_batchnorm(recognizer)
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
        return recognizer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        train_subsampling_ratio = self.cfg.get("train_subsampling_ratio", -1.0)
        if train_subsampling_ratio > 0.0:
            data_len = len(train_data)
            excluded_len = data_len - int(train_subsampling_ratio * data_len)
            train_data, split_train_data = torch.utils.data.random_split(
                train_data, (data_len - excluded_len, excluded_len)
            )

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            eval_subsampling_ratio = self.cfg.get("eval_subsampling_ratio", -1.0)
            if eval_subsampling_ratio > 0.0:
                data_len = len(val_data)
                excluded_len = data_len - int(eval_subsampling_ratio * data_len)
                val_data, excluded_dataset = torch.utils.data.random_split(
                    val_data, (data_len - excluded_len, excluded_len)
                )
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    @staticmethod
    def default_best_metric_value():
        return -torch.inf

    @staticmethod
    def default_other_metric_snapshot():
        return None


@TRAINERS.register_module("IncrSegTrainer")
class IncrSegTrainer(OpenSegTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger.info("=> Building incremental learner ...")
        self.incr_learner = self.build_incremental_learner()
        self.optimizer = self.build_incr_optimizer()
        # rebuild the scheduler after updating the optimizer.
        self.scheduler = self.build_scheduler()

    def before_epoch(self):
        if is_pytorch_model(self.incr_learner):
            self.incr_learner.train()
        if unwrap_model(self.incr_learner).need_teacher_model:
            unwrap_model(self.incr_learner).teacher_model.eval()
        super().before_epoch()

    def model_forward(self, input_dict):
        model_output = self.incr_learner(input_dict)
        return model_output

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
        if hasattr(unwrap_model(incr_learner), "need_teacher_model"):
            unwrap_model(incr_learner).inject_teacher_model(self.model)
            unwrap_model(incr_learner).teacher_model_hooks = self.model_hooks
        return incr_learner

    def build_model(self):
        # use the same model for incremental learner
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = model.cuda()

        return model

    def build_incr_optimizer(self):
        all_named_params = {
            name: param
            for name, param in self.incr_learner.named_parameters()
            if "teacher_model" not in name
        }
        return build_optimizer_from_named_params(
            self.cfg.optimizer, all_named_params, self.cfg.param_dicts
        )

    # def build_model_hook(self):
    #     return nullcontext()

    def build_recognizer(self):
        return None



# class _ModelWrapper:
#     def __init__(self, trainer):
#         self.trainer = trainer
#         self._original_model = trainer.model

#     def __call__(self, input_dict):
#         # Call the trainer's model_forward method
#         return self.trainer.model_forward(input_dict)

#     def __getattr__(self, name):
#         return getattr(self._original_model, name)

#     @classmethod
#     def __instancecheck__(cls, instance) -> bool:
#         return isinstance(instance.original_model, nn.Module)
