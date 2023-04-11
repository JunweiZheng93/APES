from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset
import os


@DATASETS.register_module()
class ModelNet(BaseDataset):

    METAINFO = dict(classes=('airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                             'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                             'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                             'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                             'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
                             'wardrobe', 'xbox'))

    # __init__ method will copy metadata, join data_root and data_prefix, and compose pipelines
    def __init__(self, data_root, data_prefix, pipeline, metainfo=METAINFO):
        super().__init__(data_root=data_root, data_prefix=data_prefix, pipeline=pipeline, metainfo=metainfo)

    def load_data_list(self):
        data_list = []
        pcd_prefix = self.data_prefix.get('pcd_path', None)  # data_prefix.__class__ == dict
        cls_label_prefix = self.data_prefix.get('cls_label_path', None)
        for pcd_name, cls_label_name in zip(sorted(os.listdir(pcd_prefix)), sorted(os.listdir(cls_label_prefix))):
            data_list.append(dict(classes=self.metainfo['classes'],
                                  pcd_path=os.path.join(pcd_prefix, pcd_name),
                                  cls_label_path=os.path.join(cls_label_prefix, cls_label_name)))
        return data_list
