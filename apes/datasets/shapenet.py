from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset
import os


@DATASETS.register_module()
class ShapeNet(BaseDataset):

    METAINFO = dict(classes=('airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                             'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table'),
                    mapping={0: (0, 1, 2, 3), 1: (4, 5), 2: (6, 7), 3: (8, 9, 10, 11), 4: (12, 13, 14, 15),
                             5: (16, 17, 18), 6: (19, 20, 21), 7: (22, 23), 8: (24, 25, 26, 27), 9: (28, 29),
                             10: (30, 31, 32, 33, 34, 35), 11: (36, 37), 12: (38, 39, 40), 13: (41, 42, 43),
                             14: (44, 45, 46), 15: (47, 48, 49)},
                    palette=((152, 223, 138), (174, 199, 232), (255, 105, 180), (31, 119, 180), (255, 187, 120),
                             (188, 189, 34), (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213),
                             (148, 103, 189), (196, 156, 148), (23, 190, 207), (186, 85, 211), (247, 182, 210),
                             (66, 188, 102), (219, 219, 141), (140, 57, 197), (202, 185, 52), (213, 92, 176),
                             (200, 54, 131), (92, 193, 61), (78, 71, 183), (172, 114, 82), (255, 127, 14),
                             (91, 163, 138), (153, 98, 156), (140, 153, 101), (158, 218, 229), (178, 127, 135),
                             (178, 127, 135), (120, 185, 128), (146, 111, 194), (44, 160, 44), (112, 128, 144),
                             (96, 207, 209), (227, 119, 194), (51, 176, 203), (94, 106, 211), (82, 84, 163),
                             (100, 85, 144), (255, 127, 80), (0, 100, 0), (173, 255, 47), (64, 224, 208),
                             (0, 255, 255), (25, 25, 112), (178, 76, 76), (255, 0, 255), (152, 223, 138)))

    # __init__ method will copy metadata, join data_root and data_prefix, and compose pipelines
    def __init__(self, data_root, data_prefix, pipeline, metainfo=METAINFO):
        super().__init__(data_root=data_root, data_prefix=data_prefix, pipeline=pipeline, metainfo=metainfo)

    def load_data_list(self):
        data_list = []
        pcd_prefix = self.data_prefix.get('pcd_path', None)  # data_prefix.__class__ == dict
        cls_label_prefix = self.data_prefix.get('cls_label_path', None)
        seg_label_prefix = self.data_prefix.get('seg_label_path', None)
        for pcd_name, cls_label_name, seg_label_name in zip(sorted(os.listdir(pcd_prefix)),
                                                            sorted(os.listdir(cls_label_prefix)),
                                                            sorted(os.listdir(seg_label_prefix))):
            data_list.append(dict(classes=self.metainfo['classes'],
                                  mapping=self.metainfo['mapping'],
                                  palette=self.metainfo['palette'],
                                  pcd_path=os.path.join(pcd_prefix, pcd_name),
                                  cls_label_path=os.path.join(cls_label_prefix, cls_label_name),
                                  seg_label_path=os.path.join(seg_label_prefix, seg_label_name)))
        return data_list
