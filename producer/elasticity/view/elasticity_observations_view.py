from packages.enums import InferenceQuality
from producer.data.resolution import Resolution


class ElasticityObservationsView:
    """
    A view class that provides read-only access to elasticity parameter observations.
    
    This class serves as a facade that provides AI agents with comprehensive
    information about the current state of elasticity parameters without
    allowing modifications.
    """

    def __init__(self, elasticity_handler):
        """
        Initialize the observations view with an ElasticityHandler instance.
        
        Args:
            elasticity_handler: The underlying ElasticityHandler instance
        """
        self._elasticity_handler = elasticity_handler

    def get_current_fps(self) -> int:
        """
        Gets the current frames per second value.

        Returns:
            int: The current FPS value
        """
        return self._elasticity_handler.state_fps.value

    def get_current_resolution(self) -> Resolution:
        """
        Gets the current resolution value.

        Returns:
            Resolution: The current resolution value
        """
        return self._elasticity_handler.state_resolution.value

    def get_current_inference_quality(self) -> InferenceQuality:
        """
        Gets the current inference quality value.

        Returns:
            InferenceQuality: The current inference quality value
        """
        return self._elasticity_handler.state_inference_quality.value