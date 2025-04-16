from producer.elasticity.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler

from producer.elasticity.slo.slo_manager import SloManager
from producer.elasticity.slo.slo_status import SloStatus


class HeuristicAgentOld(ElasticityAgent):
    """
    A rule-based agent that uses heuristics to maintain Service Level Objectives (SLOs)
    in a distributed task(video) processing system.

    This agent follows a set of predefined rules to monitor system state and take appropriate
    actions to optimize the quality of experience while ensuring computational resources
    are properly utilized.

    Key Features:
    1. SLO States (Ok, Warning, Critical)

    2. Priority-Based Decision Making:
    When all SLOs are being met, it tries to improve quality in the following order: resolution, fps, workload)
    When SLOs are in warning state, it takes measured corrective actions
    When SLOs reach critical thresholds, it takes immediate action to reduce load

    3. Resource-Specific Responses:
    For queue-related issues, it prioritizes reducing FPS first
    For memory-related issues, it prioritizes reducing resolution first
    Workload reduction is used as a general solution for both issues


    4. Oscillation Detection:
    Tracks recent actions to detect if the system is oscillating between increase and decrease
    Stabilizes by taking no action when oscillation is detected

    5. Configurable Thresholds:
    Warning and critical thresholds for both queue size and memory usage
    These can be adjusted based on your specific system requirements
    """
    # OSCILLATION_WINDOW_SIZE must be even for oscillation detection.
    OSCILLATION_WINDOW_SIZE = 2 * 2

    INCREASE_ACTIONS = {action for action in ActionType if action.name.startswith("INCREASE")}
    DECREASE_ACTIONS = {action for action in ActionType if action.name.startswith("DECREASE")}

    def __init__(self, elasticity_handler: ElasticityHandler):
        """
        Initialize the Heuristic Agent

        Args:
            elasticity_handler: The handler for changing system parameters
        """
        super().__init__(elasticity_handler)

        # Tracking previous states to detect oscillations
        self.previous_actions = []
        self.max_history = 5  # Number of previous actions to track

    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the heuristic decision process

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """

        action, success = self._select_action()

        # Update action history
        self._update_action_history(action)

        return action, success

    def reset(self):
        """Reset the agent's state"""
        self.previous_actions = []

    def _select_action(self) -> tuple[ActionType, bool]:
        """
        Select the best action based on current system state using heuristics

        Args:
            res_idx: Current resolution index
            max_res_idx: Maximum resolution index
            fps_idx: Current FPS index
            max_fps_idx: Maximum FPS index
            wl_idx: Current workload index
            max_wl_idx: Maximum workload index

        Returns:
            tuple[ActionType, bool]: Selected action and whether it was successful
        """
        qsize_slo_ratio, mem_slo_ratio = self.slo_manager.get_all_slo_ratios(track_stats=True)

        qsize_slo_status = SloManager.get_slo_status(qsize_slo_ratio)
        mem_slo_status = SloManager.get_slo_status(mem_slo_ratio)

        # Check if any SLO is in critical or warning state
        critical_slo_violation = (qsize_slo_status == SloStatus.CRITICAL or mem_slo_status == SloStatus.CRITICAL)
        warning_slo_violation = (qsize_slo_status == SloStatus.WARNING or mem_slo_status == SloStatus.WARNING)

        # If we're in critical state, take immediate action to reduce load
        if critical_slo_violation:
            print('Critical Violation')
            # If queue is the bigger issue, prioritize reducing in order: workload, fps, resolution
            if qsize_slo_ratio > mem_slo_ratio:
                if self.elasticity_handler.state_work_load.can_decrease():
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif self.elasticity_handler.state_fps.can_decrease():
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()
                elif self.elasticity_handler.state_resolution.can_decrease():
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()

            # If memory is the bigger issue, prioritize reducing in order: workload, resolution, fps
            else:
                if self.elasticity_handler.state_work_load.can_decrease():
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif self.elasticity_handler.state_resolution.can_decrease():
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()
                elif self.elasticity_handler.state_fps.can_decrease():
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()

        # If we're in warning state, take a more measured approach
        if warning_slo_violation:
            print('Warning Violation')
            # Check if we're oscillating (repeatedly increasing and decreasing)
            if self._is_oscillating():
                return ActionType.NONE, True  # Stabilize by doing nothing

            # Determine most effective action based on current state
            if qsize_slo_ratio > mem_slo_ratio:
                # Queue issues - reduce workload or FPS
                if self.elasticity_handler.state_work_load.can_decrease():
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif self.elasticity_handler.state_fps.can_decrease():
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()

            else:
                # Memory issues - reduce resolution or workload
                if self.elasticity_handler.state_work_load.can_decrease():
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif self.elasticity_handler.state_resolution.can_decrease():
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()

        if not (critical_slo_violation or warning_slo_violation):
            # If we're in a good state, try to improve quality (workload first, then resolution, then FPS)
            if self.elasticity_handler.state_resolution.can_increase():
                return ActionType.INCREASE_RESOLUTION, self.elasticity_handler.increase_resolution()
            elif self.elasticity_handler.state_fps.can_increase():
                return ActionType.INCREASE_FPS, self.elasticity_handler.increase_fps()
            elif self.elasticity_handler.state_work_load.can_increase():
                return ActionType.INCREASE_WORK_LOAD, self.elasticity_handler.increase_work_load()

        return ActionType.NONE, True # We're at maximum quality already or unable to decrease further

    def _is_oscillating(self, window_size=OSCILLATION_WINDOW_SIZE) -> bool:
        """
        Check if the system is oscillating between increasing and decreasing actions.

        Returns:
            window_size: size of the oscillation window (must be divisible by 2)
            bool: True if oscillation is detected, False otherwise.
        """
        if len(self.previous_actions) < window_size:
            return False

        recent_action = self.previous_actions[-window_size:]
        kinds = [(True if a in HeuristicAgentOld.INCREASE_ACTIONS else False) for a in recent_action]

        # Check if the pattern alternates starting with True or False
        pattern1 = [True if i % 2 == 0 else False for i in range(window_size)]
        pattern2 = [False if i % 2 == 0 else True for i in range(window_size)]

        return kinds == pattern1 or kinds == pattern2

    def _update_action_history(self, action: ActionType):
        """
        Update the history of previous actions

        Args:
            action: The action that was just taken
        """
        self.previous_actions.append(action)
        if len(self.previous_actions) > self.max_history:
            self.previous_actions.pop(0)