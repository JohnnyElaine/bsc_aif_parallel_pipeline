from packages.enums import InferenceQuality
from producer.data.resolution import Resolution
from producer.elasticity.handler.data.state import State


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

    def get_current_fps_state(self) -> State:
        """
        Gets the current frames per second state.

        Returns:
            int: The current FPS state
        """
        return self._elasticity_handler.state_fps

    def get_current_resolution_state(self) -> State:
        """
        Gets the current resolution state.

        Returns:
            Resolution: The current resolution state
        """
        return self._elasticity_handler.state_resolution

    def get_current_inference_quality_state(self) -> State:
        """
        Gets the current inference quality state.

        Returns:
            InferenceQuality: The current inference quality state
        """
        return self._elasticity_handler.state_inference_quality