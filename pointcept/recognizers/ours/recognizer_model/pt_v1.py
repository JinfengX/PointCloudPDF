import torch
from torch import nn
import pointops
from pointcept.models.builder import MODELS


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )
        return x


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

    def forward(self, backbone_features):
        p1, _, o1 = backbone_features["enc1"]["forward_getOutput"]
        p2, _, o2 = backbone_features["enc2"]["forward_getOutput"]
        p3, _, o3 = backbone_features["enc3"]["forward_getOutput"]
        p4, _, o4 = backbone_features["enc4"]["forward_getOutput"]
        p5, x5_enc, o5 = backbone_features["enc5"]["forward_getOutput"]
        x5_dec = backbone_features["dec5.1"]["forward_getOutput"][1]
        x4 = backbone_features["dec4.1"]["forward_getOutput"][1]
        x3 = backbone_features["dec3.1"]["forward_getOutput"][1]
        x2 = backbone_features["dec2.1"]["forward_getOutput"][1]
        x1 = backbone_features["dec1.1"]["forward_getOutput"][1]

        r5 = self.dec5([p5, x5_dec, o5], [p5, x5_enc, o5])
        r4 = self.dec4([p4, x4, o4], [p5, r5, o5])
        r3 = self.dec3([p3, x3, o3], [p4, r4, o4])
        r2 = self.dec2([p2, x2, o2], [p3, r3, o3])
        r1 = self.dec1([p1, x1, o1], [p2, r2, o2])
        conf = self.confidence(r1)

        return conf
