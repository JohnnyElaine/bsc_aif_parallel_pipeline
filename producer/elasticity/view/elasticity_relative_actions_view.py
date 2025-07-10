from producer.elasticity.interface.elasticity_relative_action_interface import ElasticityRelativeActionInterface


class ElasticityRelativeActionsView(ElasticityRelativeActionInterface):
    """
    A view class that provides a restricted interface to ElasticityHandler,
    exposing only the relative parameter-changing actions available to AI agents.
    
    This class serves as a facade that limits access to only the elasticity
    parameter modification methods (increase/decrease), ensuring a clean and 
    controlled interface for AI agents.
    """

    def __init__(self, elasticity_handler):
        """
        Initialize the relative actions view with an ElasticityHandler instance.
        
        Args:
            elasticity_handler: The underlying ElasticityHandler instance
        """
        self._elasticity_handler = elasticity_handler

    def increase_inference_quality(self) -> bool:
        """
        Increases the inference quality/workload to the next possible value.

        Returns:
            bool: True if the workload was increased, False if it was already at the maximum.
        """
        return self._elasticity_handler.increase_inference_quality()

    def decrease_inference_quality(self) -> bool:
        """
        Decreases the inference quality/workload to the previous possible value.

        Returns:
            bool: True if the workload was decreased, False if it was already at the minimum.
        """
        return self._elasticity_handler.decrease_inference_quality()

    def increase_fps(self) -> bool:
        """
        Increases the frames per second (FPS) to the next possible value.

        Returns:
            bool: True if the FPS was increased, False if it was already at the maximum.
        """
        return self._elasticity_handler.increase_fps()

    def decrease_fps(self) -> bool:
        """
        Decreases the frames per second (FPS) to the previous possible value.

        Returns:
            bool: True if the FPS was decreased, False if it was already at the minimum.
        """
        return self._elasticity_handler.decrease_fps()

    def increase_resolution(self) -> bool:
        """
        Increases the resolution to the next possible value.

        Returns:
            bool: True if the resolution was increased, False if it was already at the maximum.
        """
        return self._elasticity_handler.increase_resolution()

    def decrease_resolution(self) -> bool:
        """
        Decreases the resolution to the previous possible value.

        Returns:
            bool: True if the resolution was decreased, False if it was already at the minimum.
        """
        return self._elasticity_handler.decrease_resolution()
