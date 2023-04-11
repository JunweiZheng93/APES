from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
import torch


@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self, mode='val'):
        super(Accuracy, self).__init__()
        self.mode = mode

    def process(self, inputs, data_samples: list[dict]):  # data_samples is a List of Dict, not a List of ClsDataSample
        for data_sample in data_samples:
            result = dict(gt_cls_label=data_sample['gt_cls_label'], pred_cls_label=data_sample['pred_cls_label'])
            self.results.append(result)  # self.results is actually the 'results' in the compute_metrics method

    def compute_metrics(self, results) -> dict:
        # this method is for predict and tensor mode
        gt_cls_labels = torch.tensor([result['gt_cls_label'] for result in results])
        pred_cls_labels = torch.tensor([result['pred_cls_label'] for result in results])
        acc = torch.sum(pred_cls_labels == gt_cls_labels) / gt_cls_labels.shape[0]
        if self.mode == 'val':
            return dict(val_acc=acc)
        elif self.mode == 'test':
            return dict(test_acc=acc)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')

    @staticmethod
    def calculate_metrics(pred_cls_logits, gt_cls_labels_onehot) -> float:
        # this method is for loss mode, since there is no 'train_evaluator' in training api
        pred_cls_labels_prob = torch.softmax(pred_cls_logits, dim=1)
        pred_cls_labels = torch.max(pred_cls_labels_prob, dim=1)[1]  # (N, C) -> (N,)
        gt_cls_labels = torch.max(gt_cls_labels_onehot, dim=1)[1]  # (N, C) -> (N,)
        return torch.sum(pred_cls_labels == gt_cls_labels) / gt_cls_labels.shape[0]
