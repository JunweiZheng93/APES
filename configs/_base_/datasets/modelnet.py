# dataloaders
train_dataloader = dict(batch_size=8,  # batch size per GPU
                        num_workers=4,  # number of workers to load data
                        persistent_workers=True,
                        pin_memory=True,
                        drop_last=True,
                        dataset=dict(type='ModelNet',
                                     data_root='./data/modelnet',
                                     data_prefix=dict(pcd_path='pcd/train/', cls_label_path='label/train/'),
                                     pipeline=[dict(type='LoadPCD'),
                                               dict(type='LoadCLSLabel'),
                                               dict(type='ShufflePointsOrder'),
                                               dict(type='DataAugmentation', axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05),
                                               dict(type='ToCLSTensor'),
                                               dict(type='PackCLSInputs')]),
                        sampler=dict(type='DefaultSampler',  # DefaultSampler is designed for epoch-based training. It can handle both distributed and non-distributed training.
                                     shuffle=True),
                        collate_fn=dict(type='default_collate'))  # this will concatenate all the data in a batch into a single tensor
val_dataloader = dict(batch_size=8,
                      num_workers=4,
                      persistent_workers=True,
                      pin_memory=True,
                      drop_last=True,
                      dataset=dict(type='ModelNet',
                                   data_root='./data/modelnet',
                                   data_prefix=dict(pcd_path='pcd/test/', cls_label_path='label/test/'),
                                   pipeline=[dict(type='LoadPCD'),
                                             dict(type='LoadCLSLabel'),
                                             dict(type='ToCLSTensor'),
                                             dict(type='PackCLSInputs')]),
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      collate_fn=dict(type='default_collate'))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='Accuracy', mode='val')
test_evaluator = dict(type='Accuracy', mode='test')
