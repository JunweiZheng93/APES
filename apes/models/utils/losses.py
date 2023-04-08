from mmengine import MODELS
from torch import nn


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, cls_labels):
        loss = self.loss_fn(preds, cls_labels)
        return loss


@MODELS.register_module()
class ConsistencyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, tgt):
        loss_list = []
        for i in range(len(tgt)):
            for j in range(len(tgt)):
                if i < j:
                    loss_list.append(self.loss_fn(tgt[i], tgt[j]))
                else:
                    continue
        return sum(loss_list) / len(tgt)
