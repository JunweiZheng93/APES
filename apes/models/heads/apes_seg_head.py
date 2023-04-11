from torch import nn
from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class APESSegHead(BaseModule):
    def __init__(self, init_cfg=None):
        super(APESSegHead, self).__init__(init_cfg)
        self.conv1 = nn.Sequential(nn.Conv1d(2240, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Conv1d(128, 50, 1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)  # (B, 2240, 2048) -> (B, 256, 2048)
        x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x)  # (B, 256, 2048) -> (B, 128, 2048)
        x = self.conv4(x)  # (B, 128, 2048) -> (B, 50, 2048)
        return x
