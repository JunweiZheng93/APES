from mmengine.registry import MODELS
from mmengine.model import BaseModel
from typing import List
from torch import Tensor
from ...structures.seg_data_sample import SegDataSample
import torch
from einops import pack


@MODELS.register_module()
class APESSegmentor(BaseModel):
    def __init__(self,
                 backbone: dict,
                 neck: dict = None,
                 head: dict = None,
                 data_preprocessor: dict = None,
                 init_cfg: List[dict] = None):
        super(APESSegmentor, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.head = MODELS.build(head)
        self.ce_loss = MODELS.build(dict(type='CrossEntropyLoss', reduction='mean'))

    def forward(self, inputs: Tensor, data_samples: List[SegDataSample], mode: str):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.tensor(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, inputs: Tensor, data_samples: List[SegDataSample]) -> dict:
        gt_cls_labels_onehot, gt_seg_labels_onehot = self.get_gt_labels_onehot(data_samples)
        losses = dict()
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        pred_seg_logits = self.head(x)
        ce_loss = self.ce_loss(pred_seg_logits, gt_seg_labels_onehot)
        losses.update(dict(loss=ce_loss))
        return losses

    def predict(self, inputs: Tensor, data_samples: List[SegDataSample]) -> List[SegDataSample]:
        data_samples_list = []
        gt_cls_labels_onehot, gt_seg_labels_onehot = self.get_gt_labels_onehot(data_samples)
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        pred_seg_logits = self.head(x)
        pred_seg_labels_prob = torch.softmax(pred_seg_logits, dim=1)
        pred_seg_labels = torch.max(pred_seg_labels_prob, dim=1)[1]
        for data_sample, pred_seg_logit, pred_seg_label_prob, pred_seg_label in zip(data_samples, pred_seg_logits, pred_seg_labels_prob, pred_seg_labels):
            data_sample.pred_seg_logit = pred_seg_logit
            data_sample.pred_seg_label_prob = pred_seg_label_prob
            data_sample.pred_seg_label = pred_seg_label
            data_samples_list.append(data_sample)
        return data_samples_list

    def tensor(self, inputs: Tensor, data_samples: List[SegDataSample]) -> Tensor:
        gt_cls_labels_onehot, gt_seg_labels_onehot = self.get_gt_labels_onehot(data_samples)
        x = self.extract_features(inputs, gt_cls_labels_onehot)
        seg_logits = self.head(x)
        return seg_logits

    @staticmethod
    def get_gt_labels_onehot(data_samples: List[SegDataSample]):
        cls_labels_list = []
        seg_labels_list = []
        for data_sample in data_samples:
            cls_labels_list.append(data_sample.gt_cls_label_onehot)
            seg_labels_list.append(data_sample.gt_seg_label_onehot)
        cls_labels, _ = pack(cls_labels_list, '* C N')  # shape == (B, C, N=1)
        seg_labels, _ = pack(seg_labels_list, '* C N')  # shape == (B, C, N)
        return cls_labels, seg_labels

    def extract_features(self, inputs, shape_classes):
        x = self.backbone(inputs, shape_classes)
        if self.neck is not None:
            x = self.neck(x)
        return x
