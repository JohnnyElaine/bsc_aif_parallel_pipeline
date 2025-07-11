import logging

from packages.enums import InferenceQuality
from producer.data.resolution import Resolution
from producer.data.task_config import TaskConfig
from producer.elasticity.handler.data.state import State
from producer.elasticity.handler.possible_values.generation import (generate_possible_resolutions,
                                                                    generate_possible_fps,
                                                                    generate_possible_work_loads)
from producer.elasticity.interface.elasticity_absolute_action_interface import ElasticityAbsoluteActionInterface
from producer.elasticity.interface.elasticity_relative_action_interface import ElasticityRelativeActionInterface
from producer.elasticity.view.elasticity_absolute_actions_view import ElasticityAbsoluteActionsView
from producer.elasticity.view.elasticity_observations_view import ElasticityObservationsView
from producer.elasticity.view.elasticity_relative_actions_view import ElasticityRelativeActionsView
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ElasticityHandler(ElasticityAbsoluteActionInterface, ElasticityRelativeActionInterface):
    """
    A class to handle elasticity-related operations for a task, such as adjusting workload,
    frames per second (FPS), and resolution.

    This class interacts with a `TaskConfig`, `TaskGenerator`, and `RequestHandler` to
    dynamically adjust task parameters based on workload and performance requirements.

    Attributes:
        _task_generator (TaskGenerator): The task generator responsible for creating tasks.
        _request_handler (RequestHandler): The handler responsible for managing requests.
        state_resolution (State): The current state of the resolution.
        state_fps (State): The current state of the frames per second (FPS).
        state_inference_quality (State): The current state of the workload.
    """

    def __init__(self, target_config: TaskConfig, task_generator: TaskGenerator, request_handler: RequestHandler):
        self._task_generator = task_generator
        self._request_handler = request_handler

        self.state_resolution = ElasticityHandler._create_state(
            target_config.max_resolution,
            generate_possible_resolutions(target_config.max_resolution),
            self._change_resolution
        )
        self.state_fps = ElasticityHandler._create_state(
            target_config.max_fps,
            generate_possible_fps(target_config.max_fps),
            self._change_fps
        )
        self.state_inference_quality = ElasticityHandler._create_state(
            target_config.max_work_load,
            generate_possible_work_loads(target_config.max_work_load),
            self._change_inference_quality
        )

    def actions_absolute(self) -> ElasticityAbsoluteActionsView:
        """
        Returns a view that exposes only the elasticity parameter-changing actions.
        
        This method provides AI agents with a clean interface containing only
        the methods they need to modify elasticity parameters: FPS, Resolution,
        and Inference Quality.
        
        Returns:
            ElasticityAbsoluteActionsView: A restricted view with only parameter-changing methods
        """
        return ElasticityAbsoluteActionsView(self)

    def actions_relative(self) -> ElasticityRelativeActionsView:
        """
        Returns a view that exposes only the relative elasticity parameter-changing actions.
        
        This method provides AI agents with a clean interface containing only
        the methods they need to relatively modify elasticity parameters: increase/decrease
        FPS, Resolution, and Inference Quality.
        
        Returns:
            ElasticityRelativeActionsView: A restricted view with only relative parameter-changing methods
        """
        return ElasticityRelativeActionsView(self)

    def observations(self) -> ElasticityObservationsView:
        """
        Returns a view that exposes only the elasticity parameter observations.
        
        This method provides AI agents with a clean read-only interface to observe
        current parameter values, capacities, limits, and capabilities without
        allowing modifications.
        
        Returns:
            ElasticityObservationsView: A read-only view with parameter observation methods
        """
        return ElasticityObservationsView(self)

    def change_inference_quality_index(self, index: int) -> bool:
        """
        Changes the inference quality/workload parameter by index.

        Args:
            index (int): The index in the list of possible inference quality values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self.state_inference_quality.change(index)

    def change_fps_index(self, index: int) -> bool:
        """
        Changes the FPS parameter by index.

        Args:
            index (int): The index in the list of possible FPS values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self.state_fps.change(index)

    def change_resolution_index(self, index: int) -> bool:
        """
        Changes the resolution parameter by index.

        Args:
            index (int): The index in the list of possible resolution values.

        Returns:
            bool: True if the change was successful, False otherwise.
        """
        return self.state_resolution.change(index)

    def increase_resolution(self) -> bool:
        """
        Increases the resolution to the next possible value.

        Returns:
            bool: True if the resolution was increased, False if it was already at the maximum.
        """
        return self.state_resolution.increase()

    def decrease_resolution(self) -> bool:
        """
        Decreases the resolution to the previous possible value.

        Returns:
            bool: True if the resolution was decreased, False if it was already at the minimum.
        """
        return self.state_resolution.decrease()

    def increase_fps(self) -> bool:
        """
        Increases the frames per second (FPS) to the next possible value.

        Returns:
            bool: True if the FPS was increased, False if it was already at the maximum.
        """
        return self.state_fps.increase()

    def decrease_fps(self) -> bool:
        """
        Decreases the frames per second (FPS) to the previous possible value.

        Returns:
            bool: True if the FPS was decreased, False if it was already at the minimum.
        """
        return self.state_fps.decrease()

    def increase_inference_quality(self) -> bool:
        """
        Increases the workload to the next possible value.

        Returns:
            bool: True if the workload was increased, False if it was already at the maximum.
        """
        return self.state_inference_quality.increase()

    def decrease_inference_quality(self) -> bool:
        """
        Decreases the workload to the previous possible value.

        Returns:
            bool: True if the workload was decreased, False if it was already at the minimum.
        """
        return self.state_inference_quality.decrease()

    # Basic observations for current values
    def get_current_fps(self) -> int:
        """Gets the current frames per second value."""
        return self.state_fps.value

    def get_current_resolution(self) -> Resolution:
        """Gets the current resolution value."""
        return self.state_resolution.value

    def get_current_inference_quality(self) -> InferenceQuality:
        """Gets the current inference quality value."""
        return self.state_inference_quality.value

    def _change_inference_quality(self, work_load: InferenceQuality):
        """
        Propagates the change to the request handler.

        Args:
            work_load (InferenceQuality): The new workload value to be applied.
        """
        self._request_handler.change_inference_quality(work_load)

    def _change_fps(self, fps: int):
        """
        Updates Propagates the fps change to the task generator.

        Args:
            fps (int): The new frames per second (FPS) value to be applied.
        """
        self._task_generator.set_fps(fps)

    def _change_resolution(self, resolution: Resolution):
        """
        Updates Propagates the change to the task generator.

        Args:
            resolution (Resolution): The new resolution value to be applied.
        """
        self._task_generator.set_resolution(resolution)

    @staticmethod
    def _create_state(init_value, possible_states: list, change_function: callable) -> State:
        """
        Creates a new state object based on the initial value and the possible states generated by the provided function.

        Args:
            init_value: The initial value for the state.
            change_function (callable): The function to generate possible states.

        Returns:
            State: A new state object.
        """
        index = possible_states.index(init_value)
        return State(index, possible_states, change_function)