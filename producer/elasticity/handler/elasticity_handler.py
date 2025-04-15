import logging

import producer.elasticity.handler.possible_values.aspect_ratios as AllAspectRatios
from packages.enums import WorkLoad
from producer.request_handling.request_handler import RequestHandler
from producer.data.resolution import Resolution
from producer.data.task_config import TaskConfig
from producer.elasticity.handler.data.state import State
from producer.elasticity.handler.possible_values.resolutions import AllResolutions
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
        state_work_load (State): The current state of the workload.
    """

    def __init__(self, target_config: TaskConfig, task_generator: TaskGenerator, request_handler: RequestHandler):
        self._task_generator = task_generator
        self._request_handler = request_handler

        self.state_resolution = ElasticityHandler._create_state(
            target_config.max_resolution,
            ElasticityHandler._generate_possible_resolutions(target_config.max_resolution))
        self.state_fps = ElasticityHandler._create_state(
            target_config.max_fps,
            ElasticityHandler._generate_possible_fps(target_config.max_fps))
        self.state_work_load = ElasticityHandler._create_state(
            target_config.max_work_load,
            ElasticityHandler._generate_possible_work_loads(target_config.max_work_load))

        # TODO check if functions needs synchronization

    @property
    def resolution(self):
        """
        Returns the current resolution.

        Returns:
            Resolution: The current resolution value.
        """
        return self.state_resolution.value

    @property
    def max_fps(self):
        """
        Returns the current frames per second (FPS).

        Returns:
            int: The current FPS value.
        """
        return self.state_fps.max

    @property
    def fps(self):
        """
        Returns the current frames per second (FPS).

        Returns:
            int: The current FPS value.
        """
        return self.state_fps.value

    @property
    def work_load(self):
        """
        Returns the current workload.

        Returns:
            WorkLoad: The current workload value.
        """
        return self.state_work_load.value

    def queue_size(self):
        """Return the approximate size of the queue (not reliable!)."""
        return self._task_generator.queue_size()

    def increase_resolution(self):
        """
        Increases the resolution to the next possible value.

        Returns:
            bool: True if the resolution was increased, False if it was already at the maximum.
        """
        return self._increase_state(self.state_resolution, self._change_resolution)

    def decrease_resolution(self):
        """
        Decreases the resolution to the previous possible value.

        Returns:
            bool: True if the resolution was decreased, False if it was already at the minimum.
        """
        return self._decrease_state(self.state_resolution, self._change_resolution)

    def increase_fps(self):
        """
        Increases the frames per second (FPS) to the next possible value.

        Returns:
            bool: True if the FPS was increased, False if it was already at the maximum.
        """
        return self._increase_state(self.state_fps, self._change_fps)

    def decrease_fps(self):
        """
        Decreases the frames per second (FPS) to the previous possible value.

        Returns:
            bool: True if the FPS was decreased, False if it was already at the minimum.
        """
        return self._decrease_state(self.state_fps, self._change_fps)

    def increase_work_load(self):
        """
        Increases the workload to the next possible value.

        Returns:
            bool: True if the workload was increased, False if it was already at the maximum.
        """
        return self._increase_state(self.state_work_load, self._change_work_load)

    def decrease_work_load(self):
        """
        Decreases the workload to the previous possible value.

        Returns:
            bool: True if the workload was decreased, False if it was already at the minimum.
        """
        return self._decrease_state(self.state_work_load, self._change_work_load)

    def _change_work_load(self, work_load: WorkLoad):
        """
        Propagates the change to the request handler.

        Args:
            work_load (WorkLoad): The new workload value to be applied.
        """
        self._request_handler.change_work_load(work_load)

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
    def _generate_possible_work_loads(max_work_load: WorkLoad) -> list[WorkLoad]:
        """
        Generates a list of possible workload values.

        Returns:
            list[WorkLoad]: A list of possible workload values.
        """
        return [w for w in list(WorkLoad) if w.value <= max_work_load.value]

    @staticmethod
    def _generate_possible_resolutions(max_res: Resolution) -> list[Resolution]:
        """
        Generates a list of possible resolution values based on the initial configuration's aspect ratio.

        Returns:
            list[Resolution]: A list of possible resolution values.
        """
        match max_res.get_aspect_ratio():
            case AllAspectRatios.ASPECT_RATIO_16_9:
                all_resolutions = AllResolutions.RATIO_16_9
            case AllAspectRatios.ASPECT_RATIO_4_3:
                all_resolutions = AllResolutions.RATIO_4_3
            case _:
                all_resolutions = AllResolutions.RATIO_16_9

        return [res for res in all_resolutions if res.pixels <= max_res.pixels]

    @staticmethod
    def _generate_possible_fps(max_fps: int):
        """
        Returns possible FPS values from max_fps down to 10, stepping by -2.

        Returns:
            list[int]: A list of possible FPS values.
        """
        a = list(range(max_fps, 9, -2))
        a.reverse()
        return a

    @staticmethod
    def _increase_state(state: State, change_function: callable):
        """
        Increases the state to the next possible value and applies the change using the provided function.

        Args:
            state (State): The current state to be increased.
            change_function (callable): The function to apply the change.

        Returns:
            bool: True if the state was increased, False if it was already at the maximum.
        """
        if state.current_index >= len(state.possible_states) - 1:
            return False

        state.current_index += 1
        change_function(state.value)

        log.debug(f'{change_function.__name__}: {state.possible_states[state.current_index- 1]} -> {state.possible_states[state.current_index]}')

        return True

    @staticmethod
    def _decrease_state(state: State, change_function: callable):
        """
        Decreases the state to the previous possible value and applies the change using the provided function.

        Args:
            state (State): The current state to be decreased.
            change_function (callable): The function to apply the change.

        Returns:
            bool: True if the state was decreased, False if it was already at the minimum.
        """
        if state.current_index <= 0:
            return False

        state.current_index -= 1
        change_function(state.value)

        log.debug(f'{change_function.__name__}: {state.possible_states[state.current_index + 1]} -> {state.possible_states[state.current_index]}')

        return True

    @staticmethod
    def _create_state(init_value, possible_states: list):
        """
        Creates a new state object based on the initial value and the possible states generated by the provided function.

        Args:
            init_value: The initial value for the state.
            states_gen_function (callable): The function to generate possible states.

        Returns:
            State: A new state object.
        """
        index = possible_states.index(init_value)
        return State(index, possible_states)