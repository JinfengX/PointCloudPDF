import torch
from torch import nn
import torch.nn.functional as F

from pointcept.models.builder import build_model
from pointcept.models.losses.builder import build_criteria
from pointcept.utils.misc import is_parallel_model, is_pytorch_model

from ..builder import INCREMENTALLEARNER
from typing import Union


@INCREMENTALLEARNER.register_module("PointPdf-incr-v1m1")
class PointPdfIncrV1(nn.Module):
    def __init__(self, backbone=None, eval_criteria=None):
        super().__init__()
        self.need_teacher_model = True
        self.incr_backbone = build_model(backbone)
        self.criteria = IncrDistillKlLoss()
        self.eval_criteria = build_criteria(eval_criteria)

    def forward(self, input_dict):
        seg_logits = self.incr_backbone(input_dict)
        if self.training:
            teacher_seg_logits = self.get_teacher_output(input_dict)
            loss = self.criteria(
                seg_logits, teacher_seg_logits, input_dict["segment_incr"]
            )
            return dict(loss=loss)
        elif "segment" in input_dict:
            loss = self.eval_criteria(seg_logits, input_dict["segment_incr_remap"])
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)

    def get_teacher_output(self, input_dict):
        assert self.teacher_model is not None, "Teacher model is not set."
        with torch.no_grad():
            teacher_output = self.teacher_model(input_dict)
            teacher_seg_logits = self.teacher_model_hooks["backbone"]["forward_output"]
        return teacher_seg_logits

    def inject_teacher_model(
        self,
        model: Union[
            nn.Module, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel
        ],
    ):
        if not is_pytorch_model(model):
            raise TypeError("model must be a pytorch model")
        self.teacher_model = model

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = self.incr_backbone.state_dict(
            destination=destination,
            prefix=prefix + "incr_backbone.",
            keep_vars=keep_vars,
        )
        return state_dict


class IncrDistillKlLoss(nn.Module):
    def __init__(self, pred_temp=1.0, target_temp=1.0, loss_weight=1.0):
        super().__init__()
        self.pred_temp = pred_temp
        self.target_temp = target_temp
        self.loss_weight = loss_weight

    def forward(self, pred, target, segment_incr):
        pred = F.log_softmax(pred / self.pred_temp, dim=1)

        n_pts, num_incr_class = pred.shape
        valid_mask = segment_incr != -1
        incr_target = torch.cat(
            [
                torch.softmax(target / self.target_temp, dim=1),
                torch.zeros(n_pts, num_incr_class - target.shape[1]).cuda(),
            ],
            dim=1,
        )
        incr_target[valid_mask] = torch.eye(num_incr_class, device=pred.device)[segment_incr[valid_mask]]
        loss = F.kl_div(
            pred,
            incr_target,
            reduction="batchmean",
        )
        return loss * self.loss_weight
