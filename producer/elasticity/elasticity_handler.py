import logging
import producer.elasticity.possible_values.aspect_ratios as AllAspectRatios
from producer.elasticity.possible_values.work_loads import ALL_WORK_LOADS
from producer.elasticity.possible_values.resolutions import AllResolutions
from packages.enums import WorkLoad
from producer.communication.request_handler import RequestHandler
from producer.data.resolution import Resolution
from producer.data.task_config import TaskConfig
from producer.elasticity.state import State
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ElasticityHandler:
    """
    A class to handle elasticity-related operations for a task, such as adjusting workload,
    frames per second (FPS), and resolution.

    This class interacts with a `TaskConfig`, `TaskGenerator`, and `RequestHandler` to
    dynamically adjust task parameters based on workload and performance requirements.

    Attributes:
        _init_config (TaskConfig): The configuration object holding task parameters.
        _task_generator (TaskGenerator): The task generator responsible for creating tasks.
        _request_handler (RequestHandler): The handler responsible for managing requests.
        _state_resolution (State): The current state of the resolution.
        _state_fps (State): The current state of the frames per second (FPS).
        _state_work_load (State): The current state of the workload.
    """

    def __init__(self, initial_task_config: TaskConfig, task_generator: TaskGenerator, request_handler: RequestHandler):
        self._init_config = initial_task_config
        self._task_generator = task_generator
        self._request_handler = request_handler

        self._state_resolution = ElasticityHandler._create_state(initial_task_config.resolution, self._generate_possible_resolutions())
        self._state_fps = ElasticityHandler._create_state(initial_task_config.fps, self._generate_possible_fps())
        self._state_work_load = ElasticityHandler._create_state(initial_task_config.work_load, self._generate_possible_work_loads())

        # TODO check if functions needs synchronization

    @property
    def resolution(self):
        return self._state_resolution.value

    @property
    def fps(self):
        return self._state_fps.value

    @property
    def work_load(self):
        return self._state_work_load.value

    def increase_resolution(self):
        return self._increase_state(self._state_resolution, self._change_resolution)

    def decrease_resolution(self):
        return self._decrease_state(self._state_resolution, self._change_resolution)

    def increase_fps(self):
        return self._increase_state(self._state_fps, self._change_fps)

    def decrease_fps(self):
        return self._decrease_state(self._state_fps, self._change_fps)

    def increase_work_load(self):
        return self._increase_state(self._state_work_load, self._change_work_load)

    def decrease_work_load(self):
        return self._decrease_state(self._state_work_load, self._change_work_load)

    def _change_work_load(self, work_load: WorkLoad):
        self._request_handler.change_work_load(work_load)

    def _change_fps(self, fps: int):
        """
        Updates the frame per second (FPS) configuration and propagates the change
        to the task generator.

        Args:
            fps (int): The new frames per second (FPS) value to be applied.
        """
        self._task_generator.set_fps(self._init_config.fps)

    def _change_resolution(self, resolution: Resolution):
        self._task_generator.set_resolution(resolution)

    def _generate_possible_work_loads(self) -> list[WorkLoad]:
        return ALL_WORK_LOADS

    def _generate_possible_resolutions(self) -> list[Resolution]:
        src_res = self._init_config.resolution

        match src_res.get_aspect_ratio():
            case AllAspectRatios.ASPECT_RATIO_16_9:
                all_resolutions = AllResolutions.RATIO_16_9
            case AllAspectRatios.ASPECT_RATIO_4_3:
                all_resolutions = AllResolutions.RATIO_4_3
            case _:
                all_resolutions = AllResolutions.RATIO_16_9

        return [res for res in all_resolutions if res <= src_res]

    def _generate_possible_fps(self):
        return [fps for fps in range(5, self._init_config.fps + 5, 5)]

    @staticmethod
    def _increase_state(state: State, change_function: callable):
        if state.current_index >= len(state.possible_states) - 1:
            return False

        state.current_index += 1
        change_function(state.value)
        return True

    @staticmethod
    def _decrease_state(state: State, change_function: callable):
        if state.current_index <= 0:
            return False

        state.current_index -= 1
        change_function(state.value)
        return True

    @staticmethod
    def _create_state(init_value, states_gen_function: callable):
        possible_states = states_gen_function()
        index = possible_states.index(init_value)
        return State(index, possible_states)

    def _create_resolution_state(self):
        possible_resolution_states = self._generate_possible_resolutions()
        current_resolution_index = possible_resolution_states.index(self._init_config.resolution)
        return State(current_resolution_index, possible_resolution_states)

    def _create_fps_state(self):
        possible_fps_states = self._generate_possible_fps()
        current_fps_index = possible_fps_states.index(self._init_config.fps)
        return State(current_fps_index, possible_fps_states)