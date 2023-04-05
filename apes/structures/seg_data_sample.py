from mmengine.structures import BaseDataElement
from torch import Tensor


class SegDataSample(BaseDataElement):

    @property
    def gt_cls_label(self) -> Tensor:
        return self.gt_cls_label

    @gt_cls_label.setter
    def gt_cls_label(self, value: Tensor) -> None:
        self.set_field(value, 'gt_cls_label', dtype=Tensor)

    @gt_cls_label.deleter
    def gt_cls_label(self) -> None:
        del self.gt_cls_label

    @property
    def gt_seg_label(self) -> Tensor:
        return self.gt_seg_label

    @gt_seg_label.setter
    def gt_seg_label(self, value: Tensor) -> None:
        self.set_field(value, 'gt_seg_label', dtype=Tensor)

    @gt_seg_label.deleter
    def gt_seg_label(self) -> None:
        del self.gt_seg_label

    @property
    def pred_seg_label(self) -> Tensor:
        return self.pred_seg_label

    @pred_seg_label.setter
    def pred_seg_label(self, value: Tensor) -> None:
        self.set_field(value, 'pred_seg_label', dtype=Tensor)

    @pred_seg_label.deleter
    def pred_seg_label(self) -> None:
        del self.pred_seg_label

    @property
    def seg_logits(self) -> Tensor:
        return self.seg_logits

    @seg_logits.setter
    def seg_logits(self, value: Tensor) -> None:
        self.set_field(value, 'seg_logits', dtype=Tensor)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self.seg_logits
