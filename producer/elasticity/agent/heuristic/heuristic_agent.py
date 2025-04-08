from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SLOStatus
import logging

log = logging.getLogger('producer')


class HeuristicAgent(ElasticityAgent):
    """
    A rule-based agent that uses heuristics to maintain Service Level Objectives (SLOs)
    in a distributed video processing system.

    This agent follows a set of predefined rules to monitor system state and take appropriate
    actions to optimize the quality of experience while ensuring computational resources
    are properly utilized.
    """

    def __init__(self, elasticity_handler: ElasticityHandler):
        """
        Initialize the Heuristic Agent

        Args:
            elasticity_handler: The handler for changing system parameters
        """
        super().__init__(elasticity_handler)
        # Configurable parameters for the heuristic agent
        self.queue_threshold_critical = 0.9  # Critical threshold for queue SLO (90% of max)
        self.queue_threshold_warning = 0.7  # Warning threshold for queue SLO (70% of max)
        self.memory_threshold_critical = 0.9  # Critical threshold for memory SLO (90% of max)
        self.memory_threshold_warning = 0.7  # Warning threshold for memory SLO (70% of max)

        # Tracking previous states to detect oscillations
        self.previous_actions = []
        self.max_history = 5  # Number of previous actions to track

    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the heuristic decision process

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        # Get current SLO status
        queue_slo_ratio = self.slo_manager.queue_slo_ratio()
        memory_slo_ratio = self.slo_manager.memory_slo_ratio()

        # Get current system state
        current_resolution_idx = self.elasticity_handler.state_resolution.current_index
        max_resolution_idx = len(self.elasticity_handler.state_resolution.possible_states) - 1

        current_fps_idx = self.elasticity_handler.state_fps.current_index
        max_fps_idx = len(self.elasticity_handler.state_fps.possible_states) - 1

        current_workload_idx = self.elasticity_handler.state_work_load.current_index
        max_workload_idx = len(self.elasticity_handler.state_work_load.possible_states) - 1

        # Log current state
        log.debug(f"Current state - Queue: {queue_slo_ratio:.2f}, Memory: {memory_slo_ratio:.2f}, "
                  f"Resolution: {current_resolution_idx}/{max_resolution_idx}, "
                  f"FPS: {current_fps_idx}/{max_fps_idx}, "
                  f"Workload: {current_workload_idx}/{max_workload_idx}")

        # Determine action based on heuristics
        action, success = self._select_action(
            queue_slo_ratio, memory_slo_ratio,
            current_resolution_idx, max_resolution_idx,
            current_fps_idx, max_fps_idx,
            current_workload_idx, max_workload_idx
        )

        # Update action history
        self._update_action_history(action)

        return action, success

    def _select_action(self, queue_ratio, memory_ratio,
                       res_idx, max_res_idx,
                       fps_idx, max_fps_idx,
                       wl_idx, max_wl_idx) -> tuple[ActionType, bool]:
        """
        Select the best action based on current system state using heuristics

        Args:
            queue_ratio: Current queue size ratio (relative to SLO)
            memory_ratio: Current memory usage ratio (relative to SLO)
            res_idx: Current resolution index
            max_res_idx: Maximum resolution index
            fps_idx: Current FPS index
            max_fps_idx: Maximum FPS index
            wl_idx: Current workload index
            max_wl_idx: Maximum workload index

        Returns:
            tuple[ActionType, bool]: Selected action and whether it was successful
        """
        # Check if any SLO is in critical state
        critical_slo_violation = (queue_ratio > self.queue_threshold_critical or
                                  memory_ratio > self.memory_threshold_critical)

        # Check if any SLO is in warning state
        warning_slo_violation = (queue_ratio > self.queue_threshold_warning or
                                 memory_ratio > self.memory_threshold_warning)

        # If we're in a good state, try to improve quality (workload first, then resolution, then FPS)
        if not warning_slo_violation:
            # Try to increase quality in priority order: workload, resolution, FPS
            if wl_idx < max_wl_idx:
                return ActionType.INCREASE_WORK_LOAD, self.elasticity_handler.increase_work_load()
            elif res_idx < max_res_idx:
                return ActionType.INCREASE_RESOLUTION, self.elasticity_handler.increase_resolution()
            elif fps_idx < max_fps_idx:
                return ActionType.INCREASE_FPS, self.elasticity_handler.increase_fps()
            else:
                return ActionType.DO_NOTHING, True  # We're at maximum quality already

        # If we're in critical state, take immediate action to reduce load
        if critical_slo_violation:
            # Determine which resource is stressed more
            queue_is_worse = queue_ratio > memory_ratio

            # If queue is the bigger issue, prioritize reducing FPS first
            if queue_is_worse:
                if fps_idx > 0:
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()
                elif wl_idx > 0:
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif res_idx > 0:
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()
            # If memory is the bigger issue, prioritize reducing resolution first
            else:
                if res_idx > 0:
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()
                elif wl_idx > 0:
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
                elif fps_idx > 0:
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()

        # If we're in warning state, take a more measured approach
        if warning_slo_violation:
            # Check if we're oscillating (repeatedly increasing and decreasing)
            if self._is_oscillating():
                return ActionType.DO_NOTHING, True  # Stabilize by doing nothing

            # Determine most effective action based on current state
            if queue_ratio > memory_ratio:
                # Queue issues - reduce workload or FPS
                if fps_idx > 0:
                    return ActionType.DECREASE_FPS, self.elasticity_handler.decrease_fps()
                elif wl_idx > 0:
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()
            else:
                # Memory issues - reduce resolution or workload
                if res_idx > 0:
                    return ActionType.DECREASE_RESOLUTION, self.elasticity_handler.decrease_resolution()
                elif wl_idx > 0:
                    return ActionType.DECREASE_WORK_LOAD, self.elasticity_handler.decrease_work_load()

        # Default action if no other conditions are met
        return ActionType.DO_NOTHING, True

    def _is_oscillating(self) -> bool:
        """
        Check if the system is oscillating between increasing and decreasing actions

        Returns:
            bool: True if oscillation is detected, False otherwise
        """
        if len(self.previous_actions) < 4:
            return False

        # Look for patterns like increase->decrease->increase->decrease
        increase_actions = {ActionType.INCREASE_RESOLUTION, ActionType.INCREASE_FPS, ActionType.INCREASE_WORK_LOAD}
        decrease_actions = {ActionType.DECREASE_RESOLUTION, ActionType.DECREASE_FPS, ActionType.DECREASE_WORK_LOAD}

        # Check last 4 actions
        recent_actions = self.previous_actions[-4:]

        # Pattern 1: increase -> decrease -> increase -> decrease
        pattern1 = (recent_actions[0] in increase_actions and
                    recent_actions[1] in decrease_actions and
                    recent_actions[2] in increase_actions and
                    recent_actions[3] in decrease_actions)

        # Pattern 2: decrease -> increase -> decrease -> increase
        pattern2 = (recent_actions[0] in decrease_actions and
                    recent_actions[1] in increase_actions and
                    recent_actions[2] in decrease_actions and
                    recent_actions[3] in increase_actions)

        return pattern1 or pattern2

    def _update_action_history(self, action: ActionType):
        """
        Update the history of previous actions

        Args:
            action: The action that was just taken
        """
        self.previous_actions.append(action)
        if len(self.previous_actions) > self.max_history:
            self.previous_actions.pop(0)

    def reset(self):
        """Reset the agent's state"""
        self.previous_actions = []