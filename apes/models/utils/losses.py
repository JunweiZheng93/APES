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
