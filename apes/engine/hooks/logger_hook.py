from mmengine.registry import HOOKS
from mmengine.hooks import LoggerHook
from typing import Optional, Sequence, Union
DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


@HOOKS.register_module()
class ModifiedLoggerHook(LoggerHook):
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """get rid of the annoying log of 'Exp name: xxx' after every epoch"""
        if self.every_n_train_iters(runner, self.interval_exp_name):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and not self.ignore_last):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)


    def after_test_epoch(self,
                         runner,
                         metrics=None):
        """don't save metrics to json file after test"""
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), 'test', with_non_scalar=True)
        runner.logger.info(log_str)
