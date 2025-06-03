from functools import partial

import torch
from ..builder import RECOGNIZER


@RECOGNIZER.register_module()
class MaxProbability(object):
    def __init__(self, method=None):
        if method == "msp":
            self.prob_func = self.msp
        elif method == "max_logits":
            self.prob_func = self.ml
        else:
            raise ValueError(f"Unknown MaxProbability method {method}")

    def __call__(self, input_dict):
        seg_logits = self.model_hooks["backbone"]["forward_output"]
        score = -self.prob_func(seg_logits)
        return dict(score=score)

    def msp(self, seg_logits):
        msp = seg_logits.log_softmax(dim=-1)
        msp = msp.max(dim=-1)[0]
        return msp

    def ml(self, seg_logits):
        ml = seg_logits.max(dim=-1)[0]
        return ml
