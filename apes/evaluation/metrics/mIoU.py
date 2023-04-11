import numpy as np
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class InstanceMeanIoU(BaseMetric):
    def __init__(self, mode='val'):
        super(InstanceMeanIoU, self).__init__()
        self.mode = mode

    def process(self, inputs, data_samples: list[dict]):  # data_samples is a List of Dict, not a List of SegDataSample
        for data_sample in data_samples:
            shape_id = int(data_sample['gt_cls_label'].cpu().numpy())
            part_ids = self.dataset_meta['mapping'][shape_id]
            pred_seg_label = data_sample['pred_seg_label'].cpu().numpy().astype(np.uint8)
            gt_seg_label = data_sample['gt_seg_label'].cpu().numpy().astype(np.uint8)
            part_ious = []
            for part_id in part_ids:
                I = np.sum(np.logical_and(pred_seg_label == part_id, gt_seg_label == part_id))
                U = np.sum(np.logical_or(pred_seg_label == part_id, gt_seg_label == part_id))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            self.results.append(np.mean(part_ious))

    def compute_metrics(self, results) -> dict:
        if self.mode == 'val':
            return dict(val_instance_mIoU=np.mean(results))
        elif self.mode == 'test':
            return dict(test_instance_mIoU=np.mean(results))
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')


@METRICS.register_module()
class CategoryMeanIoU(BaseMetric):
    def __init__(self, mode='val'):
        super(CategoryMeanIoU, self).__init__()
        self.mode = mode

    def process(self, inputs, data_samples):
        pass

    def compute_metrics(self, results: list[dict]) -> dict:
        pass
