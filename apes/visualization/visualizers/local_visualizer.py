from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS


@VISUALIZERS.register_module()
class APESVisualizer(Visualizer):
    def add_image(self, name, pcd) -> None:
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, pcd)
