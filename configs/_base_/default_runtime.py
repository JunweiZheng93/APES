default_scope = 'mmengine'  # default scope for all registries
work_dir = './work_dirs'  # working dir to save ckpts and logs
load_from = None  # load from which checkpoint
resume = False  # whether to resume training from the loaded checkpoint
launcher = 'none'  # supported launchers are 'pytorch', 'mpi', 'slurm' and ‘none’. ‘none’ is for single GPU training
experiment_name = '{{fileBasenameNoExtension}}'  # use cfg file name as exp name. if not specified, timestamp will be used as experiment_name
env_cfg = dict(cudnn_benchmark=True,  # whether to enable cudnn benchmark
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # set multi process parameters
               dist_cfg=dict(backend='nccl', init_method='env://'))  # set distributed parameters
randomness = dict(seed=None,  # seed
                  diff_rank_seed=False,  # whether to use different seed for different ranks
                  deterministic=False)  # if True, cudnn_benchmark in env_cfg will be forced to False by mmengine
log_level = 'INFO'  # set log level
log_processor = dict(by_epoch=True,  # whether to log by epoch or by iteration
                     custom_cfg=[dict(data_src='loss',  # original data source
                                      log_name='loss',  # you can rename the data source here
                                      method_name='mean',  # method to aggregate data. can be 'mean', 'max', 'min' and 'current'
                                      window_size='epoch'),
                                 dict(data_src='acc',
                                      log_name='acc',
                                      method_name='mean',
                                      window_size='epoch')])
visualizer = dict(type='APESVisualizer',
                  vis_backends=[dict(type='WandbVisBackend',
                                     log_code_name=f'{experiment_name}',  # wandb code name
                                     init_kwargs=dict(entity='kit-cvhci',  # wandb username or team name
                                                      project='APES',  # wandb project name
                                                      name=f'{experiment_name}'))])  # wandb run name
# all other configs
cfg = dict(compile=True,  # can be a boolean or a dict containing compile options. only valid in PyTorch 2.x
           sync_bn='torch',  # whether to use sync bn in multi-GPU case. can be 'torch', 'mmcv' and None
           find_unused_parameters=False)  # whether to find unused parameters in model
