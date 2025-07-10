from abc import ABC, abstractmethod
from packages.enums import InferenceQuality
from producer.data.resolution import Resolution


class ElasticityAbsoluteActionInterface(ABC):
    """
    Abstract interface defining the available elasticity parameter actions for AI agents.
    
    This interface provides a clear contract for what elasticity parameters can be
    modified by an AI agent, including FPS, Resolution, and Inference Quality.
    """
        
    @abstractmethod
    def change_inference_quality_index(self, index: int) -> bool:
        """
        Changes the inference quality/workload parameter by index.

        Args:
            index (int): The index in the list of possible inference quality values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        pass

    @abstractmethod
    def change_fps_index(self, index: int) -> bool:
        """
        Changes the FPS parameter by index.

        Args:
            index (int): The index in the list of possible FPS values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        pass

    @abstractmethod
    def change_resolution_index(self, index: int) -> bool:
        """
        Changes the resolution parameter by index.

        Args:
            index (int): The index in the list of possible resolution values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        pass
