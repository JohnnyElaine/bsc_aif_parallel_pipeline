import logging
from typing import Optional

from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.slo.slo_util import SloUtil
from producer.elasticity.view.heuristic_agent_observations import HeuristicAgentObservations
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class HeuristicAgent(ElasticityAgent):
    """
    A heuristic-based agent that manages system elasticity to maintain SLOs while maximizing
    stream quality parameters (FPS, resolution, workload) in a balanced way.

    The agent decides actions based on the current state of all 4 SLOs and avoids oscillations
    by implementing cooldown periods and tracking historical actions.
    """

    # Class constants
    CAPACITY_EQUALITY_THRESHOLD = 0.01
    WARNING_TREND_THRESHOLD = 0.05
    IMPROVEMENT_TREND_THRESHOLD = 0.02
    COOLDOWN_CYCLES = 2
    SLO_HISTORY_SIZE = 5
    MAX_CONSECUTIVE_ACTIONS = 2

    # Action opposites mapping
    ACTION_OPPOSITES = {
        GeneralActionType.INCREASE_FPS: GeneralActionType.DECREASE_FPS,
        GeneralActionType.DECREASE_FPS: GeneralActionType.INCREASE_FPS,
        GeneralActionType.INCREASE_RESOLUTION: GeneralActionType.DECREASE_RESOLUTION,
        GeneralActionType.DECREASE_RESOLUTION: GeneralActionType.INCREASE_RESOLUTION,
        GeneralActionType.INCREASE_INFERENCE_QUALITY: GeneralActionType.DECREASE_INFERENCE_QUALITY,
        GeneralActionType.DECREASE_INFERENCE_QUALITY: GeneralActionType.INCREASE_INFERENCE_QUALITY
    }

    def __init__(self, elasticity_handler: ElasticityHandler, request_handler: RequestHandler, task_generator: TaskGenerator, track_slo_stats=True):
        super().__init__(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)

        # Create observations and actions views for clean interface
        self.observations = HeuristicAgentObservations(
            elasticity_handler.observations(),
            self._slo_manager
        )
        self.actions = elasticity_handler.actions_relative()

        # Cooldown management to prevent oscillations
        self.cooldown_counter = 0
        self.last_action_type = None
        self.consecutive_action_count = 0

        # Track SLO history to detect trends - now for all 4 SLOs
        self.queue_slo_value_history = []
        self.memory_slo_value_history = []
        self.global_processing_slo_value_history = []
        self.worker_processing_slo_value_history = []

        # Decision thresholds for proactive adjustments
        self.upscale_threshold = SloUtil.WARNING_THRESHOLD

        log.info("Heuristic Elasticity Agent initialized with all 4 SLOs")

    def step(self) -> tuple[GeneralActionType, bool]:
        """
        Perform a single step of the agent's decision process.

        Returns:
            tuple[GeneralActionType, bool]: The action taken and whether it was successful
        """
        # Get current SLO values for all 4 SLOs
        queue_slo_value, memory_slo_value, global_processing_slo_value, worker_processing_slo_value = self.observations.get_all_slo_values(track_stats=True)
        queue_status, memory_status, global_processing_status, worker_processing_status = self.observations.get_all_slo_status()

        # Update SLO history for all 4 SLOs
        self.queue_slo_value_history.append(queue_slo_value)
        self.memory_slo_value_history.append(memory_slo_value)
        self.global_processing_slo_value_history.append(global_processing_slo_value)
        self.worker_processing_slo_value_history.append(worker_processing_slo_value)
        
        # Keep history size manageable
        for history in [self.queue_slo_value_history, self.memory_slo_value_history, 
                       self.global_processing_slo_value_history, self.worker_processing_slo_value_history]:
            if len(history) > self.SLO_HISTORY_SIZE:
                history.pop(0)

        # Check if we're in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return GeneralActionType.NONE, True

        # Handle CRITICAL SLO states first (immediate action required)
        if (queue_status == SloStatus.CRITICAL or memory_status == SloStatus.CRITICAL or 
            global_processing_status == SloStatus.CRITICAL or worker_processing_status == SloStatus.CRITICAL):
            log.warning(f"Current state - Queue SLO: {queue_status.name} ({queue_slo_value:.3f}), "
                      f"Memory SLO: {memory_status.name} ({memory_slo_value:.3f}), "
                      f"Global Processing SLO: {global_processing_status.name} ({global_processing_slo_value:.3f}), "
                      f"Worker Processing SLO: {worker_processing_status.name} ({worker_processing_slo_value:.3f})")

            action, success = self._handle_critical_slo()
            if success:
                self._update_action_tracking(action)
                return action, success

        # If all SLOs are OK, check if we can improve quality
        if (queue_status == SloStatus.OK and memory_status == SloStatus.OK and 
            global_processing_status == SloStatus.OK and worker_processing_status == SloStatus.OK):
            action, success = self._try_quality_improvement()
            if success:
                self._update_action_tracking(action)
                return action, success

        # Handle WARNING SLO states (Do nothing for now, but could be extended)

        return GeneralActionType.NONE, True


    def _handle_critical_slo(self) -> tuple[GeneralActionType, bool]:
        """
        Handle CRITICAL SLO states with immediate balanced action across parameters.

        Returns:
            tuple[GeneralActionType, bool]: Action taken and whether it was successful
        """
        # Get current capacities using the observations view
        res_capacity = self.observations.get_resolution_capacity()
        fps_capacity = self.observations.get_fps_capacity()
        workload_capacity = self.observations.get_inference_quality_capacity()

        # Find which parameter has the highest capacity (most room to decrease)
        capacities = [
            (res_capacity, GeneralActionType.DECREASE_RESOLUTION, self.observations.can_decrease_resolution()),
            (fps_capacity, GeneralActionType.DECREASE_FPS, self.observations.can_decrease_fps()),
            (workload_capacity, GeneralActionType.DECREASE_INFERENCE_QUALITY, self.observations.can_decrease_inference_quality())
        ]

        # Filter to only those that can be decreased
        decreasable = [(cap, action) for cap, action, can_decrease in capacities if can_decrease]

        # all quality parameters are at minimum
        if not decreasable:
            return GeneralActionType.NONE, True

        # Sort by capacity (highest first)
        decreasable.sort(key=lambda x: x[0], reverse=True)

        # Check if all capacities are relatively even
        if self._are_capacities_even([cap for cap, _ in decreasable]):
            # If capacities are even, follow the specified order
            priority_order = [
                GeneralActionType.DECREASE_INFERENCE_QUALITY,
                GeneralActionType.DECREASE_FPS,
                GeneralActionType.DECREASE_RESOLUTION
            ]

            for action in priority_order:
                for _, decreasable_action in decreasable:
                    if decreasable_action == action:
                        return self._execute_action(decreasable_action)
        else:
            # Target the parameter with highest capacity
            _, action = decreasable[0]
            return self._execute_action(action)

        return GeneralActionType.NONE, True

    def _handle_warning_slo(self) -> tuple[GeneralActionType, bool]:
        """
        Handle WARNING SLO states with proactive balanced adjustments.

        Returns:
            tuple[GeneralActionType, bool]: Action taken and whether it was successful
        """
        # Check if SLO trends are getting worse for all 4 SLOs
        queue_trend = self._calculate_trend(self.queue_slo_value_history)
        memory_trend = self._calculate_trend(self.memory_slo_value_history)
        global_processing_trend = self._calculate_trend(self.global_processing_slo_value_history)
        worker_processing_trend = self._calculate_trend(self.worker_processing_slo_value_history)

        # If trends show worsening conditions, take proactive action
        if (queue_trend > self.WARNING_TREND_THRESHOLD or memory_trend > self.WARNING_TREND_THRESHOLD or 
            global_processing_trend > self.WARNING_TREND_THRESHOLD or worker_processing_trend > self.WARNING_TREND_THRESHOLD):
            log.debug(f"Worsening SLO trends detected - Queue: {queue_trend:.3f}, Memory: {memory_trend:.3f}, "
                     f"Global Processing: {global_processing_trend:.3f}, Worker Processing: {worker_processing_trend:.3f}")

            # Similar to critical handling but with lower urgency
            return self._handle_critical_slo()

        # No concerning trends, so no action needed yet
        return GeneralActionType.NONE, True

    def _try_quality_improvement(self) -> tuple[GeneralActionType, bool]:
        """
        Try to improve quality parameters in a balanced way when all SLOs are well below thresholds.

        Returns:
            tuple[GeneralActionType, bool]: Action taken and whether it was successful
        """
        # Get current SLO values for all 4 SLOs
        queue_value, memory_value, global_processing_value, worker_processing_value = self.observations.get_all_slo_values()

        # Only try to improve if all SLOs are comfortably below threshold
        if max(queue_value, memory_value, global_processing_value, worker_processing_value) > self.upscale_threshold:
            return GeneralActionType.NONE, True

        # Make sure we've been stable for a while before trying to improve
        if len(self.queue_slo_value_history) < self.SLO_HISTORY_SIZE:
            return GeneralActionType.NONE, True

        # Check if trends are stable or improving for all 4 SLOs
        queue_trend = self._calculate_trend(self.queue_slo_value_history)
        memory_trend = self._calculate_trend(self.memory_slo_value_history)
        global_processing_trend = self._calculate_trend(self.global_processing_slo_value_history)
        worker_processing_trend = self._calculate_trend(self.worker_processing_slo_value_history)

        if (queue_trend > self.IMPROVEMENT_TREND_THRESHOLD or memory_trend > self.IMPROVEMENT_TREND_THRESHOLD or 
            global_processing_trend > self.IMPROVEMENT_TREND_THRESHOLD or worker_processing_trend > self.IMPROVEMENT_TREND_THRESHOLD):
            # SLOs are trending worse, don't try to improve quality yet
            return GeneralActionType.NONE, True

        log.debug("All SLOs stable or improving - attempting quality improvement")

        # Get current capacities using the observations view
        res_capacity = self.observations.get_resolution_capacity()
        fps_capacity = self.observations.get_fps_capacity()
        workload_capacity = self.observations.get_inference_quality_capacity()

        # Find which parameter has the lowest capacity (most room to increase)
        capacities = [
            (res_capacity, GeneralActionType.INCREASE_RESOLUTION, self.observations.can_increase_resolution()),
            (fps_capacity, GeneralActionType.INCREASE_FPS, self.observations.can_increase_fps()),
            (workload_capacity, GeneralActionType.INCREASE_INFERENCE_QUALITY, self.observations.can_increase_inference_quality())
        ]

        # Filter to only those that can be increased
        increasable = [(cap, action) for cap, action, can_increase in capacities if can_increase]

        if not increasable:
            log.debug("All parameters already at maximum quality")
            return GeneralActionType.NONE, True

        # Sort by capacity (lowest first)
        increasable.sort()

        # Check if all capacities are relatively even
        if self._are_capacities_even([cap for cap, _ in increasable]):
            # If capacities are even, follow the specified order
            priority_order = [
                GeneralActionType.INCREASE_RESOLUTION,
                GeneralActionType.INCREASE_FPS,
                GeneralActionType.INCREASE_INFERENCE_QUALITY
            ]

            for action in priority_order:
                for _, increasable_action in increasable:
                    if increasable_action == action:
                        return self._execute_action(increasable_action)
        else:
            # Target the parameter with lowest capacity
            _, action = increasable[0]
            return self._execute_action(action)

    def _execute_action(self, action: GeneralActionType) -> tuple[GeneralActionType, bool]:
        """
        Execute the given action and check if it would cause oscillation.

        Args:
            action: The action to execute

        Returns:
            tuple[GeneralActionType, bool]: The action taken and whether it was successful
        """
        # Check if this would cause oscillation
        if action == self._get_opposite_action(self.last_action_type):
            log.debug(f"Skipping {action.name} to avoid oscillation")
            return GeneralActionType.NONE, True

        # Check if we've done this action too many times in a row
        if action == self.last_action_type and self.consecutive_action_count >= self.MAX_CONSECUTIVE_ACTIONS:
            log.debug(f"Skipping {action.name} to avoid consecutive repetition")
            return GeneralActionType.NONE, True

        # Execute the action using the relative actions view
        success = False
        if action == GeneralActionType.DECREASE_RESOLUTION:
            success = self.actions.decrease_resolution()
        elif action == GeneralActionType.DECREASE_FPS:
            success = self.actions.decrease_fps()
        elif action == GeneralActionType.DECREASE_INFERENCE_QUALITY:
            success = self.actions.decrease_inference_quality()
        elif action == GeneralActionType.INCREASE_RESOLUTION:
            success = self.actions.increase_resolution()
        elif action == GeneralActionType.INCREASE_FPS:
            success = self.actions.increase_fps()
        elif action == GeneralActionType.INCREASE_INFERENCE_QUALITY:
            success = self.actions.increase_inference_quality()

        if success:
            log.debug(f"Action taken: {action.name}")
        return action, success

    def _are_capacities_even(self, capacities: list) -> bool:
        """
        Check if all capacities are relatively even.

        Args:
            capacities: List of capacity values

        Returns:
            bool: True if all capacities are within threshold difference of each other
        """
        if not capacities:
            return True

        min_capacity = min(capacities)
        max_capacity = max(capacities)

        return max_capacity - min_capacity <= self.CAPACITY_EQUALITY_THRESHOLD

    def _update_action_tracking(self, action: GeneralActionType) -> None:
        """
        Update the action tracking to prevent oscillations and repetitive actions.

        Args:
            action: The action that was just taken
        """
        self.cooldown_counter = self.COOLDOWN_CYCLES

        if action == self.last_action_type:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1

        self.last_action_type = action

    def _calculate_trend(self, history: list) -> float:
        """
        Calculate the trend in a time series of SLO ratios.

        Args:
            history: List of historical SLO ratio values

        Returns:
            float: Trend value (positive means worsening, negative means improving)
        """
        if len(history) < 2:
            return 0.0

        # Simple linear trend calculation (average of differences)
        differences = [history[i] - history[i - 1] for i in range(1, len(history))]
        return sum(differences) / len(differences)

    def _get_opposite_action(self, action: Optional[GeneralActionType]) -> Optional[GeneralActionType]:
        """
        Get the opposite action for a given action type.

        Args:
            action: The action to find the opposite for

        Returns:
            GeneralActionType: The opposite action, or None if input is None or NONE
        """
        if action is None or action == GeneralActionType.NONE:
            return None

        return self.ACTION_OPPOSITES.get(action)