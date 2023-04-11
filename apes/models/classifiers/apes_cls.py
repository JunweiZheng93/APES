from mmengine.registry import MODELS, METRICS
from mmengine.model import BaseModel
from typing import List
from torch import Tensor
from ...structures.cls_data_sample import ClsDataSample
import torch
from einops import pack


@MODELS.register_module()
class APESClassifier(BaseModel):
    def __init__(self,
                 backbone: dict,
                 neck: dict = None,
                 head: dict = None,
                 data_preprocessor: dict = None,
                 init_cfg: List[dict] = None):
        super(APESClassifier, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.head = MODELS.build(head)
        self.ce_loss = MODELS.build(dict(type='CrossEntropyLoss', reduction='mean'))
        self.acc = METRICS.build(dict(type='Accuracy'))

    def forward(self, inputs: Tensor, data_samples: List[ClsDataSample], mode: str):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.tensor(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> dict:
        gt_cls_labels_onehot = self.get_gt_cls_labels_onehot(data_samples)
        losses = dict()
        x = self.extract_features(inputs)
        pred_cls_logits = self.head(x)
        ce_loss = self.ce_loss(pred_cls_logits, gt_cls_labels_onehot)
        acc = self.acc.calculate_metrics(pred_cls_logits, gt_cls_labels_onehot)
        losses.update(dict(loss=ce_loss))
        losses.update(dict(acc=acc))
        return losses

    def predict(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> List[ClsDataSample]:
        data_samples_list = []
        x = self.extract_features(inputs)
        pred_cls_logits = self.head(x)
        pred_cls_labels_prob = torch.softmax(pred_cls_logits, dim=1)
        pred_cls_labels = torch.max(pred_cls_labels_prob, dim=1)[1]
        for data_sample, pred_cls_logit, pred_cls_label_prob, pred_cls_label in zip(data_samples, pred_cls_logits, pred_cls_labels_prob, pred_cls_labels):
            data_sample.pred_cls_logit = pred_cls_logit
            data_sample.pred_cls_label_prob = pred_cls_label_prob
            data_sample.pred_cls_label = pred_cls_label
            data_samples_list.append(data_sample)
        return data_samples_list

    def tensor(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> Tensor:
        x = self.extract_features(inputs)
        cls_logits = self.head(x)
        return cls_logits

    @staticmethod
    def get_gt_cls_labels_onehot(data_samples: List[ClsDataSample]) -> Tensor:
        labels_list = []
        for data_sample in data_samples:
            assert data_sample.gt_cls_label_onehot is not None
            labels_list.append(data_sample.gt_cls_label_onehot)
        labels, ps = pack(labels_list, '* C')  # shape == (B, C)
        return labels

    def extract_features(self, inputs: Tensor) -> Tensor:
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x
