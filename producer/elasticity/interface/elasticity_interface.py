from abc import ABC, abstractmethod
from packages.enums import InferenceQuality
from producer.data.resolution import Resolution


class ElasticityInterface(ABC):
    """
    Abstract interface defining the available elasticity parameter actions for AI agents.
    
    This interface provides a clear contract for what elasticity parameters can be
    modified by an AI agent, including FPS, Resolution, and Inference Quality.
    """

    @abstractmethod
    def change_inference_quality(self, work_load: InferenceQuality) -> None:
        """
        Changes the inference quality/workload parameter.

        Args:
            work_load (InferenceQuality): The new inference quality value to be applied.
        """
        pass

    @abstractmethod
    def change_fps(self, fps: int) -> None:
        """
        Changes the frames per second parameter.

        Args:
            fps (int): The new frames per second (FPS) value to be applied.
        """
        pass

    @abstractmethod
    def change_resolution(self, resolution: Resolution) -> None:
        """
        Changes the resolution parameter.

        Args:
            resolution (Resolution): The new resolution value to be applied.
        """
        pass
