from mmengine.registry import TRANSFORMS
from .basetransform import BaseTransform
from typing import Dict
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F


@TRANSFORMS.register_module()
class ToCLSTensor(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        results['pcd'] = rearrange(torch.tensor(results['pcd']).to(torch.float32), 'N C -> C N')  # PyTorch requires (C, N) format
        results['cls_label'] = torch.tensor(results['cls_label']).to(torch.float32)   # array to tensor
        results['cls_label_onehot'] = F.one_hot(results['cls_label'].long(), 40).to(torch.float32)  # shape == (40,)
        return results


@TRANSFORMS.register_module()
class ToSEGTensor(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        results['pcd'] = rearrange(torch.tensor(results['pcd']).to(torch.float32), 'N C -> C N')  # PyTorch requires (C, N) format
        results['cls_label'] = torch.tensor(results['cls_label']).to(torch.float32)  # array to tensor
        results['cls_label_onehot'] = F.one_hot(results['cls_label'].long(), 16).to(torch.float32)  # shape == (1, 16)
        results['cls_label_onehot'] = repeat(results['cls_label_onehot'], 'C -> C 1')  # shape == (16, 1)
        results['seg_label'] = torch.tensor(results['seg_label']).to(torch.float32)  # array to tensor
        results['seg_label_onehot'] = F.one_hot(results['seg_label'].long(), 50).to(torch.float32)  # shape == (N, 50)
        results['seg_label_onehot'] = rearrange(results['seg_label_onehot'], 'N C -> C N')  # shape == (50, N)
        return results


@TRANSFORMS.register_module()
class ShufflePointsOrder(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        idx = np.random.choice(results['pcd'].shape[0], results['pcd'].shape[0], replace=False)
        results['pcd'] = results['pcd'][idx]
        if 'seg_label' in results:
            results['seg_label'] = results['seg_label'][idx]
        return results


@TRANSFORMS.register_module()
class DataAugmentation(BaseTransform):
    def __init__(self, axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05):
        super().__init__()
        jitter = Jitter(sigma, clip)
        rotation = Rotation(axis, angle)
        translation = Translation(shift)
        anisotropic_scaling = AnisotropicScaling(min_scale, max_scale)
        self.aug_list = [jitter, rotation, translation, anisotropic_scaling]

    def transform(self, results: Dict) -> Dict:
        results = np.random.choice(self.aug_list)(results)
        return results


class Jitter(BaseTransform):
    def __init__(self, sigma=0.01, clip=0.05):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        npts, nfeats = pcd.shape
        jit_pts = np.clip(self.sigma * np.random.randn(npts, nfeats), -self.clip, self.clip)
        jit_pts += pcd
        results['pcd'] = jit_pts
        return results


class Rotation(BaseTransform):
    def __init__(self, axis='y', angle=15):
        super().__init__()
        self.axis = axis
        self.angle = angle

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        angle = np.random.uniform(-self.angle, self.angle)
        angle = np.pi * angle / 180
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        if self.axis == 'x':
            rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
        elif self.axis == 'y':
            rotation_matrix = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
        elif self.axis == 'z':
            rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
        else:
            raise ValueError(f'axis should be one of x, y and z, but got {self.axis}!')
        rotated_pts = pcd @ rotation_matrix
        results['pcd'] = rotated_pts
        return results


class Translation(BaseTransform):
    def __init__(self, shift=0.2):
        super().__init__()
        self.shift = shift

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        npts = pcd.shape[0]
        x_translation = np.random.uniform(-self.shift, self.shift)
        y_translation = np.random.uniform(-self.shift, self.shift)
        z_translation = np.random.uniform(-self.shift, self.shift)
        x = np.full(npts, x_translation)
        y = np.full(npts, y_translation)
        z = np.full(npts, z_translation)
        translation = np.stack([x, y, z], axis=-1)
        translation_pts = pcd + translation
        results['pcd'] = translation_pts
        return results


class AnisotropicScaling(BaseTransform):
    def __init__(self, min_scale=0.66, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        x_factor = np.random.uniform(self.min_scale, self.max_scale)
        y_factor = np.random.uniform(self.min_scale, self.max_scale)
        z_factor = np.random.uniform(self.min_scale, self.max_scale)
        scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
        scaled_pts = pcd @ scale_matrix
        results['pcd'] = scaled_pts
        return results
