class ElasticityAbsoluteActionsView:
    """
    A view class that provides a restricted interface to ElasticityHandler,
    exposing only the index-based parameter-changing actions available to AI agents.
    
    This class serves as a facade that limits access to only the elasticity
    parameter modification methods using index-based changes, ensuring a clean 
    and controlled interface for AI agents.
    """

    def __init__(self, elasticity_handler):
        """
        Initialize the actions view with an ElasticityHandler instance.
        
        Args:
            elasticity_handler: The underlying ElasticityHandler instance
        """
        self._elasticity_handler = elasticity_handler

    def change_inference_quality_index(self, index: int) -> bool:
        """
        Changes the inference quality/workload parameter by index.

        Args:
            index (int): The index in the list of possible inference quality values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self._elasticity_handler.change_inference_quality_index(index)

    def change_fps_index(self, index: int) -> bool:
        """
        Changes the FPS parameter by index.

        Args:
            index (int): The index in the list of possible FPS values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self._elasticity_handler.change_fps_index(index)

    def change_resolution_index(self, index: int) -> bool:
        """
        Changes the resolution parameter by index.

        Args:
            index (int): The index in the list of possible resolution values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self._elasticity_handler.change_resolution_index(index)
