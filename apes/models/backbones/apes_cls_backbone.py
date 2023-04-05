from torch import nn
from einops import pack
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample


@MODELS.register_module()
class APESClsBackbone(BaseModule):
    def __init__(self, which_ds, init_cfg=None):
        super(APESClsBackbone, self).__init__(init_cfg)
        self.embedding = Embedding()
        if which_ds == 'global':
            self.ds1 = GlobalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = GlobalDownSample(512)  # 1024 pts -> 512 pts
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = LocalDownSample(512)  # 1024 pts -> 512 pts
        else:
            raise NotImplementedError
        self.n2p_attention = N2PAttention()
        self.n2p_attention1 = N2PAttention()
        self.n2p_attention2 = N2PAttention()
        self.conv = nn.Conv1d(128, 1024, 1)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        res_link_list = []
        x = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        x = self.n2p_attention(x)  # (B, 128, 2048) -> (B, 128, 2048)
        res_link_list.append(self.conv(x).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        x = self.ds1(x)  # (B, 128, 2048) -> (B, 128, 1024)
        x = self.n2p_attention1(x)  # (B, 128, 1024) -> (B, 128, 1024)
        res_link_list.append(self.conv1(x).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)
        x = self.ds2(x)  # (B, 128, 1024) -> (B, 128, 512)
        x = self.n2p_attention2(x)  # (B, 128, 512) -> (B, 128, 512)
        res_link_list.append(self.conv2(x).max(dim=-1)[0])  # (B, 128, 512) -> (B, 1024, 512) -> (B, 1024)
        x, ps = pack(res_link_list, 'B *')  # (B, 3072)
        return x
