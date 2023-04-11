from mmengine.structures import BaseDataElement
from torch import Tensor


class SegDataSample(BaseDataElement):

    @property
    def gt_cls_label(self) -> Tensor:
        return self._gt_cls_label

    @gt_cls_label.setter
    def gt_cls_label(self, value: Tensor) -> None:
        self.set_field(value, '_gt_cls_label', dtype=Tensor)

    @gt_cls_label.deleter
    def gt_cls_label(self) -> None:
        del self._gt_cls_label

    @property
    def gt_cls_label_onehot(self) -> Tensor:
        return self._gt_cls_label_onehot

    @gt_cls_label_onehot.setter
    def gt_cls_label_onehot(self, value: Tensor) -> None:
        self.set_field(value, '_gt_cls_label_onehot', dtype=Tensor)

    @gt_cls_label_onehot.deleter
    def gt_cls_label_onehot(self) -> None:
        del self._gt_cls_label_onehot

    @property
    def gt_seg_label(self) -> Tensor:
        return self._gt_seg_label

    @gt_seg_label.setter
    def gt_seg_label(self, value: Tensor) -> None:
        self.set_field(value, '_gt_seg_label', dtype=Tensor)

    @gt_seg_label.deleter
    def gt_seg_label(self) -> None:
        del self._gt_seg_label

    @property
    def gt_seg_label_onehot(self) -> Tensor:
        return self._gt_seg_label_onehot

    @gt_seg_label_onehot.setter
    def gt_seg_label_onehot(self, value: Tensor) -> None:
        self.set_field(value, '_gt_seg_label_onehot', dtype=Tensor)

    @gt_seg_label_onehot.deleter
    def gt_seg_label_onehot(self) -> None:
        del self._gt_seg_label_onehot

    @property
    def pred_seg_label(self) -> Tensor:
        return self._pred_seg_label

    @pred_seg_label.setter
    def pred_seg_label(self, value: Tensor) -> None:
        self.set_field(value, '_pred_seg_label', dtype=Tensor)

    @pred_seg_label.deleter
    def pred_seg_label(self) -> None:
        del self._pred_seg_label

    @property
    def pred_seg_logit(self) -> Tensor:
        return self._pred_seg_logit

    @pred_seg_logit.setter
    def pred_seg_logit(self, value: Tensor) -> None:
        self.set_field(value, '_pred_seg_logit', dtype=Tensor)

    @pred_seg_logit.deleter
    def pred_seg_logit(self) -> None:
        del self._pred_seg_logit

    @property
    def pred_seg_label_prob(self) -> Tensor:
        return self._pred_seg_label_prob

    @pred_seg_label_prob.setter
    def pred_seg_label_prob(self, value: Tensor) -> None:
        self.set_field(value, '_pred_seg_label_prob', dtype=Tensor)

    @pred_seg_label_prob.deleter
    def pred_seg_label_prob(self) -> None:
        del self._pred_seg_label_prob
