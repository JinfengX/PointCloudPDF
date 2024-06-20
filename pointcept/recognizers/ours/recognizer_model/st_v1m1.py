from torch import nn
import pointops
from pointcept.models.builder import MODELS


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels)
        )
        self.linear2 = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels)
        )

    def forward(
        self, feats, xyz, support_xyz, offset, support_offset, support_feats=None
    ):
        feats = self.linear1(support_feats) + pointops.interpolation(
            xyz, support_xyz, self.linear2(feats), offset, support_offset
        )
        return feats, support_xyz, support_offset


@MODELS.register_module("ST-v1m1-Recognizer")
class STRecognizer(nn.Module):
    def __init__(self, up_k, channels, num_layers):
        super().__init__()
        self.upsamples = nn.ModuleList(
            [
                Upsample(up_k, channels[i], channels[i - 1])
                for i in range(num_layers - 1, 0, -1)
            ]
        )
        self.confidence = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 1),
        )

    def forward(self, backbone_features):
        in_feats, out_feats = [], []
        in_feats.append(backbone_features["upsamples.0"]["forward_getInput"])
        in_feats.append(backbone_features["upsamples.1"]["forward_getInput"])
        in_feats.append(backbone_features["upsamples.2"]["forward_getInput"])
        in_feats.append(backbone_features["upsamples.3"]["forward_getInput"])
        out_feats.append(backbone_features["upsamples.0"]["forward_getOutput"])
        out_feats.append(backbone_features["upsamples.1"]["forward_getOutput"])
        out_feats.append(backbone_features["upsamples.2"]["forward_getOutput"])
        out_feats.append(backbone_features["upsamples.3"]["forward_getOutput"])

        feats = in_feats[0][0]
        for i, upsample in enumerate(self.upsamples):
            feats, _, _ = upsample(
                feats,
                in_feats[i][1],
                in_feats[i][2],
                in_feats[i][3],
                in_feats[i][4],
                out_feats[i][0],
            )

        conf = self.confidence(feats)
        return conf
