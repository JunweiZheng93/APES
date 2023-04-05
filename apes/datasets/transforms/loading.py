from mmengine.registry import TRANSFORMS
from .basetransform import BaseTransform
from typing import Dict
import numpy as np


@TRANSFORMS.register_module()
class LoadPCD(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        pcd_path = results['pcd_path']
        pcd = np.load(pcd_path)
        results['pcd'] = pcd
        return results


@TRANSFORMS.register_module()
class LoadCLSLabel(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        cls_label_path = results['cls_label_path']
        label = np.load(cls_label_path)
        results['cls_label'] = label
        return results


@TRANSFORMS.register_module()
class LoadSEGLabel(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        seg_label_path = results['seg_label_path']
        label = np.load(seg_label_path)
        results['seg_label'] = label
        return results
