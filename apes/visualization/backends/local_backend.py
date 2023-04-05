from mmengine.visualization.vis_backend import LocalVisBackend
from mmengine.registry import VISBACKENDS


@VISBACKENDS.register_module()
class ModifiedLocalVisBackend(LocalVisBackend):

    """this backend won't save config file and any metric values. it is used for saving images only."""

    def add_config(self, config, **kwargs):
        pass

    def add_scalar(self, name, value, step=0, **kwargs):
        pass

    def add_scalars(self, scalar_dict, step=0, file_path=None, **kwargs):
        pass
