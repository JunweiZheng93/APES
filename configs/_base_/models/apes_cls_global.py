model = dict(type='APESClassifier',
             backbone=dict(type='APESClsBackbone', which_ds='global'),
             neck=None,
             head=dict(type='APESClsHead'),
             data_preprocessor=None,  # this is used for pre-processing data in batch
             init_cfg=None)  # this is used for weight initialization

data_preprocessor = None  # this data_preprocessor will be passed to runner and overwrite the one in model
