from abc import ABC, abstractmethod


class ElasticityRelativeInterface(ABC):
    """
    Abstract interface defining the available relative elasticity parameter actions for AI agents.
    
    This interface provides a clear contract for what elasticity parameters can be
    modified relatively (increase/decrease) by an AI agent, including FPS, Resolution, 
    and Inference Quality.
    """

    @abstractmethod
    def increase_inference_quality(self) -> bool:
        """
        Increases the inference quality/workload to the next possible value.

        Returns:
            bool: True if the workload was increased, False if it was already at the maximum.
        """
        pass

    @abstractmethod
    def decrease_inference_quality(self) -> bool:
        """
        Decreases the inference quality/workload to the previous possible value.

        Returns:
            bool: True if the workload was decreased, False if it was already at the minimum.
        """
        pass

    @abstractmethod
    def increase_fps(self) -> bool:
        """
        Increases the frames per second (FPS) to the next possible value.

        Returns:
            bool: True if the FPS was increased, False if it was already at the maximum.
        """
        pass

    @abstractmethod
    def decrease_fps(self) -> bool:
        """
        Decreases the frames per second (FPS) to the previous possible value.

        Returns:
            bool: True if the FPS was decreased, False if it was already at the minimum.
        """
        pass

    @abstractmethod
    def increase_resolution(self) -> bool:
        """
        Increases the resolution to the next possible value.

        Returns:
            bool: True if the resolution was increased, False if it was already at the maximum.
        """
        pass

    @abstractmethod
    def decrease_resolution(self) -> bool:
        """
        Decreases the resolution to the previous possible value.

        Returns:
            bool: True if the resolution was decreased, False if it was already at the minimum.
        """
        pass
