from mmengine.hooks import CheckpointHook
import os.path as osp
from mmengine.fileio import FileClient, get_file_backend
from mmengine.registry import HOOKS
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from mmengine.dist import master_only


@HOOKS.register_module()
class ModifiedCheckpointHook(CheckpointHook):
    def before_train(self, runner) -> None:
        """instead of saving checkpoints to work_dir, save checkpoints to log_dir"""
        if self.out_dir is None:
            self.out_dir = runner._log_dir
        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)
        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client
        if self.out_dir != runner._log_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_ckpt' not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info(
                        'best_ckpt')
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key_indicator}'
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                            best_ckpt_name)


    def _save_best_checkpoint(self, runner, metrics) -> None:
        """get rid of the annoying stdout message"""
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_ckpt_key = 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and \
               self.file_client.isfile(best_ckpt_path) and \
               is_main_process():
                self.file_client.remove(best_ckpt_path)

            best_ckpt_name = f'best_{key_indicator}_{ckpt_filename}'
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_client.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(runtime_best_ckpt_key,
                                               self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[
                    key_indicator] = self.file_client.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(
                    runtime_best_ckpt_key,
                    self.best_ckpt_path_dict[key_indicator])
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                by_epoch=False,
                backend_args=self.backend_args)


    @master_only  # need this decorator to avoid multiple processes saving the same checkpoint
    def _publish_model(self, runner, ckpt_path: str) -> None:
        from mmengine.runner import save_checkpoint
        from mmengine.runner.checkpoint import _load_checkpoint
        checkpoint = _load_checkpoint(ckpt_path)
        assert self.published_keys is not None
        removed_keys = []
        for key in list(checkpoint.keys()):
            if key not in self.published_keys:
                removed_keys.append(key)
                checkpoint.pop(key)
        if removed_keys:
            print_log(
                f'Key {removed_keys} will be removed because they are not '
                'found in published_keys. If you want to keep them, '
                f'please set `{removed_keys}` in published_keys',
                logger='current')
        final_path = osp.splitext(ckpt_path)[0] + '.pth'  # overwrite the original checkpoint
        save_checkpoint(checkpoint, final_path)
        print_log(
            f'The checkpoint ({ckpt_path}) is published to '
            f'{final_path}.',
            logger='current')
