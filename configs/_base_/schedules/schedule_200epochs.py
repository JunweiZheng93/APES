# optimizer. check https://pytorch.org/docs/stable/optim.html#algorithms for more optimizers
optim_wrapper = dict(type='AmpOptimWrapper',  # can be OptimWrapper or AmpOptimWrapper
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1))

# mmengine will scale the lr according to the ratio=(per-GPU_bs * world_size) / base_batch_size
auto_scale_lr = dict(base_batch_size=8, enable=False)

# lr scheduler. check https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for more lr schedulers
param_scheduler = [dict(type='LinearLR',
                        start_factor=1e-4,  # start_lr = start_factor * lr, where lr is defined in optimizer
                        end_factor=1,  # end_lr = end_factor * lr, where lr is defined in optimizer
                        by_epoch=True,
                        begin=0,
                        end=10),
                   dict(type='CosineAnnealingLR',
                        T_max=190,  # 1/2 period of cosine function
                        eta_min=0,  # min lr
                        by_epoch=True,
                        begin=10,
                        end=200)]

# train/val/test loop settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop', fp16=False)
test_cfg = dict(type='TestLoop', fp16=False)

# hooks
default_hooks = dict(runtime_info=dict(type='RuntimeInfoHook'),  # update runtime information into message hub
                     timer=dict(type='IterTimerHook'),  # update the time spent during iteration into message hub
                     sampler_seed=dict(type='DistSamplerSeedHook'),  # ensure distributed Sampler shuffle is active
                     param_scheduler=dict(type='ParamSchedulerHook'),  # update some hyper-parameters of optimizer
                     logger=dict(type='ModifiedLoggerHook',  # collect logs from different components
                                 log_metric_by_epoch=True,  # this parameter only valid for TensorboardVisBackend
                                 interval=1e9,  # logging interval (every k iterations)
                                 ignore_last=False,  # ignore the log of last iteration in each epoch
                                 interval_exp_name=0),  # interval for logging experiment name. 0 means no logging
                     checkpoint=dict(type='ModifiedCheckpointHook',  # save checkpoints periodically
                                     by_epoch=True,
                                     interval=-1,  # interval for saving checkpoints (by epoch or by iteration). -1 means no saving
                                     max_keep_ckpts=-1,  # maximum checkpoints to keep. -1 means unlimited
                                     save_optimizer=True,  # whether to save optimizer status. will be deleted if not included in published_keys
                                     save_param_scheduler=True,  # whether to save param scheduler status. will be deleted if not included in published_keys
                                     published_keys=['state_dict'],  # keys to be saved in checkpoint, e.g. 'state_dict', 'optimizer', 'param_schedulers', 'meta', 'message_hub'
                                     save_last=False,  # save the last checkpoint
                                     save_best=['val_acc'],  # save best ckpts according to metrics. can be multiple metrics
                                     rule=['greater']))  # rule for best score. should have the same length as save_best
custom_hooks = None  # hooks to execute custom actions like visualizing images processed by pipeline
