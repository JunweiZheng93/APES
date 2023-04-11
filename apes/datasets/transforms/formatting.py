from mmengine.registry import TRANSFORMS
from .basetransform import BaseTransform
from typing import Dict
from ...structures.cls_data_sample import ClsDataSample
from ...structures.seg_data_sample import SegDataSample


@TRANSFORMS.register_module()
class PackCLSInputs(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        packed_results = dict()
        packed_results['inputs'] = results['pcd']  # pack inputs
        data_sample = ClsDataSample()
        data_sample.gt_cls_label_onehot = results['cls_label_onehot']
        data_sample.gt_cls_label = results['cls_label']  # pack data
        metainfo = dict()
        metainfo['classes'] = results['classes']
        metainfo['pcd_path'] = results['pcd_path']
        metainfo['cls_label_path'] = results['cls_label_path']
        data_sample.set_metainfo(metainfo)  # pack metainfo
        packed_results['data_samples'] = data_sample  # pack data_samples
        return packed_results


@TRANSFORMS.register_module()
class PackSEGInputs(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        packed_results = dict()
        packed_results['inputs'] = results['pcd']  # pack inputs
        data_sample = SegDataSample()
        data_sample.gt_cls_label_onehot = results['cls_label_onehot']
        data_sample.gt_cls_label = results['cls_label']  # pack data
        data_sample.gt_seg_label_onehot = results['seg_label_onehot']
        data_sample.gt_seg_label = results['seg_label']
        metainfo = dict()
        metainfo['classes'] = results['classes']
        metainfo['mapping'] = results['mapping']
        metainfo['palette'] = results['palette']
        metainfo['pcd_path'] = results['pcd_path']
        metainfo['cls_label_path'] = results['cls_label_path']
        metainfo['seg_label_path'] = results['seg_label_path']
        data_sample.set_metainfo(metainfo)  # pack metainfo
        packed_results['data_samples'] = data_sample  # pack data_samples
        return packed_results
