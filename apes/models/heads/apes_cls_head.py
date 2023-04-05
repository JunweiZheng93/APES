from torch import nn
from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class APESClsHead(BaseModule):
    def __init__(self, init_cfg=None):
        super(APESClsHead, self).__init__(init_cfg)
        self.linear1 = nn.Sequential(nn.Linear(3072, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)  # (B, 3072) -> (B, 1024)
        x = self.dp1(x)  # (B, 1024) -> (B, 1024)
        x = self.linear2(x)  # (B, 1024) -> (B, 256)
        x = self.dp2(x)  # (B, 256) -> (B, 256)
        x = self.linear3(x)  # (B, 256) -> (B, 40)
        return x
