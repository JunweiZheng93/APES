_base_ = ['../_base_/models/apes_seg_global.py',
          '../_base_/datasets/shapenet.py',
          '../_base_/schedules/schedule_200epochs.py',
          '../_base_/default_runtime.py']

experiment_name = '{{fileBasenameNoExtension}}'  # use cfg file name as exp name
work_dir = f'./work_dirs/{experiment_name}'  # working dir to save ckpts and logs
visualizer = dict(vis_backends=[dict(type='ModifiedLocalVisBackend')])
default_hooks = dict(checkpoint=dict(save_best=['val_instance_mIoU']))
log_processor = dict(custom_cfg=[dict(data_src='loss', log_name='loss', method_name='mean', window_size='epoch')])
