import apes  # this line is necessary because we need to register all apes modules
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    runner = Runner.from_cfg(cfg)
    os.system(f'rm -rf {os.path.join(runner.work_dir, f"{cfg.experiment_name}.py")}')  # remove cfg file from work_dir
    cfg.dump(os.path.join(runner.log_dir, f'{cfg.experiment_name}.py'))  # save cfg file to log_dir
    runner.train()


if __name__ == '__main__':
    main()
