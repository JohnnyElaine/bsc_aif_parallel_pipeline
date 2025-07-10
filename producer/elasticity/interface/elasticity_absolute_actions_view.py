from packages.enums import InferenceQuality
from producer.data.resolution import Resolution
from producer.elasticity.interface.elasticity_interface import ElasticityInterface


class ElasticityAbsoluteActionsView(ElasticityInterface):
    """
    A view class that provides a restricted interface to ElasticityHandler,
    exposing only the parameter-changing actions available to AI agents.
    
    This class serves as a facade that limits access to only the elasticity
    parameter modification methods, ensuring a clean and controlled interface
    for AI agents.
    """

    def __init__(self, elasticity_handler):
        """
        Initialize the actions view with an ElasticityHandler instance.
        
        Args:
            elasticity_handler: The underlying ElasticityHandler instance
        """
        self._elasticity_handler = elasticity_handler

    def change_inference_quality(self, work_load: InferenceQuality) -> None:
        """
        Changes the inference quality/workload parameter.

        Args:
            work_load (InferenceQuality): The new inference quality value to be applied.
        """
        self._elasticity_handler.change_inference_quality(work_load)

    def change_fps(self, fps: int) -> None:
        """
        Changes the frames per second parameter.

        Args:
            fps (int): The new frames per second (FPS) value to be applied.
        """
        self._elasticity_handler.change_fps(fps)

    def change_resolution(self, resolution: Resolution) -> None:
        """
        Changes the resolution parameter.

        Args:
            resolution (Resolution): The new resolution value to be applied.
        """
        self._elasticity_handler.change_resolution(resolution)
