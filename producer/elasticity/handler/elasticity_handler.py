import logging

from packages.enums import InferenceQuality
from producer.elasticity.handler.possible_values.generation import (generate_possible_resolutions, generate_possible_fps,
                                                                    generate_possible_work_loads)
from producer.elasticity.handler.stream_parameters import StreamParameters
from producer.elasticity.interface.actions import Actions
from producer.elasticity.interface.observations import Observations
from producer.request_handling.request_handler import RequestHandler
from producer.data.resolution import Resolution
from producer.data.task_config import TaskConfig
from producer.elasticity.handler.data.state import State
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ElasticityHandler:
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
            self.change_resolution
        )
        self.state_fps = ElasticityHandler._create_state(
            target_config.max_fps,
            generate_possible_fps(target_config.max_fps),
            self.change_fps
        )
        self.state_inference_quality = ElasticityHandler._create_state(
            target_config.max_work_load,
            generate_possible_work_loads(target_config.max_work_load),
            self.change_inference_quality
        )

        self.stream_parameters = StreamParameters(self.state_resolution, self.state_fps, self.state_inference_quality)
        # TODO refactor class to use stream_parameters, check if code can be exported to StreamParameters


    def actions(self) -> Actions:
        """
        Returns: Actions only view
        """
        return Actions()

    def change_inference_quality(self, work_load: InferenceQuality):
        """
        Propagates the change to the request handler.

        Args:
            work_load (InferenceQuality): The new workload value to be applied.
        """
        self._request_handler.change_inference_quality(work_load)

    def change_fps(self, fps: int):
        """
        Updates Propagates the fps change to the task generator.

        Args:
            fps (int): The new frames per second (FPS) value to be applied.
        """
        self._task_generator.set_fps(fps)

    def change_resolution(self, resolution: Resolution):
        """
        Updates Propagates the change to the task generator.

        Args:
            resolution (Resolution): The new resolution value to be applied.
        """
        self._task_generator.set_resolution(resolution)

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

    @staticmethod
    def _create_state(init_value, possible_states: list, change_function: callable) -> State:
        """
        Creates a new state object based on the initial value and the possible states generated by the provided function.

        Args:
            init_value: The initial value for the state.
            states_gen_function (callable): The function to generate possible states.

        Returns:
            State: A new state object.
        """
        index = possible_states.index(init_value)
        return State(index, possible_states, change_function)