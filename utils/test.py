import apes  # this line is necessary because we need to register all apes modules
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='model checkpoint file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('-vis', action='store_true', help='visualize the results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.visualizer.vis_backends = [dict(type='ModifiedLocalVisBackend')]
    if args.vis:
        if cfg.test_dataloader.dataset.type == 'ModelNet':
            cfg.custom_hooks = [dict(type='CLSVisualizationHook')]
        elif cfg.test_dataloader.dataset.type == 'ShapeNet':
            cfg.custom_hooks = [dict(type='SEGVisualizationHook')]
    runner = Runner.from_cfg(cfg)
    os.system(f'rm -rf {os.path.join(runner.work_dir, f"{cfg.experiment_name}.py")}')  # remove cfg file from work_dir
    cfg.dump(os.path.join(runner.log_dir, f'{cfg.experiment_name}.py'))  # save cfg file to log_dir
    runner.test()


if __name__ == '__main__':
    main()
