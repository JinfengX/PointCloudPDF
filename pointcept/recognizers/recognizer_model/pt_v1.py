import torch
from torch import nn
import pointops
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer import TransitionUp, Bottleneck


@MODELS.register_module("PointTransformer-Recognizer")
class PTRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        planes = [32, 64, 128, 256, 512]
        self.dec5 = TransitionUp(planes[4], planes[4])
        self.dec4 = TransitionUp(planes[4], planes[3])
        self.dec3 = TransitionUp(planes[3], planes[2])
        self.dec2 = TransitionUp(planes[2], planes[1])
        self.dec1 = TransitionUp(planes[1], planes[0])
        self.confidence = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], 1),
        )

    def forward(self, model_hooks):
        p1, _, o1 = model_hooks["backbone.enc1"]["forward_output"]
        p2, _, o2 = model_hooks["backbone.enc2"]["forward_output"]
        p3, _, o3 = model_hooks["backbone.enc3"]["forward_output"]
        p4, _, o4 = model_hooks["backbone.enc4"]["forward_output"]
        p5, x5_enc, o5 = model_hooks["backbone.enc5"]["forward_output"]
        x5_dec = model_hooks["backbone.dec5.1"]["forward_output"][1]
        x4 = model_hooks["backbone.dec4.1"]["forward_output"][1]
        x3 = model_hooks["backbone.dec3.1"]["forward_output"][1]
        x2 = model_hooks["backbone.dec2.1"]["forward_output"][1]
        x1 = model_hooks["backbone.dec1.1"]["forward_output"][1]

        r5 = self.dec5([p5, x5_dec, o5], [p5, x5_enc, o5])
        r4 = self.dec4([p4, x4, o4], [p5, r5, o5])
        r3 = self.dec3([p3, x3, o3], [p4, r4, o4])
        r2 = self.dec2([p2, x2, o2], [p3, r3, o3])
        r1 = self.dec1([p1, x1, o1], [p2, r2, o2])
        conf = self.confidence(r1)

        return conf
