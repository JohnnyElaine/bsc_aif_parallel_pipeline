import numpy as np
from pymdp.agent import Agent
import pymdp.utils as utils
from enum import Enum

from producer.elasticity.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_manager import SloManager

class ActionType(Enum):
    DO_NOTHING = 0
    INCREASE_RESOLUTION = 1
    DECREASE_RESOLUTION = 2
    INCREASE_FPS = 3
    DECREASE_FPS = 4
    INCREASE_WORK_LOAD = 5
    DECREASE_WORK_LOAD = 6

class SLOStatus(Enum):
    SATISFIED = 0
    UNSATISFIED = 1

class ActiveInferenceAgent:
    """
    Active Inference Agent that uses the Free Energy Principle to maintain
    Service Level Objectives (SLOs) in a distributed video processing system.

    The agent monitors the system's state and takes actions to optimize the quality
    of experience while ensuring computational resources are properly utilized.
    """

    def __init__(
            self,
            elasticity_handler: ElasticityHandler,
            starting_resolution_index: int,
            starting_fps_index: int,
            starting_work_load_index: int,
            task_queue_size: int,
            target_fps: int,
            target_resolution,
            planning_horizon: int = 2
    ):
        """
        Initialize the Active Inference Agent

        Args:
            elasticity_handler: The handler for changing system parameters
            starting_resolution_index: The index of the starting resolution in the possible states
            starting_fps_index: The index of the starting fps in the possible states
            starting_work_load_index: The index of the starting workload in the possible states
            task_queue_size: Current size of the task queue
            target_fps: The source video fps
            target_resolution: The source video resolution
            fps_tolerance: Minimum acceptable fps ratio compared to source (default: 0.9)
            resolution_tolerance: Minimum acceptable resolution ratio compared to source (default: 0.9)
            planning_horizon: Number of time steps to plan ahead (default: 2)
        """
        self.elasticity_handler = elasticity_handler
        self.possible_resolutions = elasticity_handler.state_resolution.possible_states
        self.possible_fps = elasticity_handler.state_fps.possible_states
        self.possible_work_loads = elasticity_handler.state_work_load.possible_states

        self.elasticity_handler.state_resolution.current_index = starting_resolution_index
        self.elasticity_handler.state_fps.current_index = starting_fps_index
        self.elasticity_handler.state_work_load.current_index = starting_work_load_index

        self.task_queue_size = task_queue_size
        self.target_fps = target_fps
        self.target_resolution = target_resolution

        self.slo_manager = SloManager(self.elasticity_handler, target_fps, target_resolution)

        # Define the dimensions of various observations
        self.num_resolution_states = len(self.possible_resolutions)
        self.num_fps_states = len(self.possible_fps)
        self.num_work_load_states = len(self.possible_work_loads)
        self.num_queue_states = 2  # Queue is either OK or too large

        # Define actions
        self.actions = list(ActionType)
        self.num_actions = len(self.actions)

        # Create the agent's generative model
        self._setup_generative_model(planning_horizon)

    def _setup_generative_model(self, planning_horizon: int):
        """
        Set up the generative model for active inference

        Args:
            planning_horizon: Number of time steps to plan ahead
        """
        # Define observation dimensions
        self.obs_dims = [
            self.num_resolution_states,  # Resolution state
            self.num_fps_states,  # FPS state
            self.num_work_load_states,  # Work load state
            self.num_queue_states,  # Queue size status (OK or too large)
            2,  # FPS SLO status (satisfied or not)
            2,  # Resolution SLO status (satisfied or not)
        ]

        # Define state dimensions (hidden states)
        self.state_dims = [
            self.num_resolution_states,  # Resolution state
            self.num_fps_states,  # FPS state
            self.num_work_load_states,  # Work load state
        ]

        self.control_dims = [self.num_actions]

        self.A = self._construct_A_matrix()
        self.B = self._construct_B_matrix()
        self.C = self._construct_C_matrix()
        self.D = self._construct_D_matrix()

        # Initialize the agent
        self.agent = Agent(A=self.A, B=self.B, C=self.C, D=self.D)

    def _construct_A_matrix(self):
        """Construct the A matrix (observation model) - Likelihood mapping from hidden states to observations"""
        # Initialize the A matrix with all zeros
        A = utils.obj_array(len(self.obs_dims))

        # For each observation modality, create a mapping from hidden states
        for obs_idx in range(len(self.obs_dims)):
            A[obs_idx] = np.zeros(self.obs_dims[obs_idx:obs_idx + 1] + self.state_dims)

        # Observation 0: Resolution state - direct mapping
        for i in range(self.num_resolution_states):
            A[0][i, i, :, :] = 1.0

        # Observation 1: FPS state - direct mapping
        for i in range(self.num_fps_states):
            A[1][i, :, i, :] = 1.0

        # Observation 2: Work load state - direct mapping
        for i in range(self.num_work_load_states):
            A[2][i, :, :, i] = 1.0

        # Observation 3: Queue size status - depends on FPS and Work load
        # Higher FPS and higher work load increase probability of large queue
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    # Queue size probability depends on FPS and work load
                    # Higher FPS or higher work load increases probability of large queue
                    queue_ok_prob = 1.0 - (0.1 * fps_idx / (self.num_fps_states - 1) +
                                           0.2 * wl_idx / (self.num_work_load_states - 1) +
                                           0.1 * res_idx / (self.num_resolution_states - 1))
                    queue_ok_prob = max(0.1, min(0.9, queue_ok_prob))  # Bound between 0.1 and 0.9

                    A[3][SLOStatus.SATISFIED.value, res_idx, fps_idx, wl_idx] = queue_ok_prob
                    A[3][SLOStatus.UNSATISFIED.value, res_idx, fps_idx, wl_idx] = 1.0 - queue_ok_prob

        # Observation 4: FPS SLO status - depends on current FPS vs source FPS
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    if self.slo_manager.fps_slo_satisfied():
                        A[4][SLOStatus.SATISFIED.value, res_idx, fps_idx, wl_idx] = 0.9
                        A[4][SLOStatus.UNSATISFIED.value, res_idx, fps_idx, wl_idx] = 0.1
                    else:
                        A[4][SLOStatus.SATISFIED.value, res_idx, fps_idx, wl_idx] = 0.1
                        A[4][SLOStatus.UNSATISFIED.value, res_idx, fps_idx, wl_idx] = 0.9

        # Observation 5: Resolution SLO status - depends on current resolution vs source resolution
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    if self.slo_manager.resolution_slo_satisfied():
                        A[5][SLOStatus.SATISFIED.value, res_idx, fps_idx, wl_idx] = 0.9
                        A[5][SLOStatus.UNSATISFIED.value, res_idx, fps_idx, wl_idx] = 0.1
                    else:
                        A[5][SLOStatus.SATISFIED.value, res_idx, fps_idx, wl_idx] = 0.1
                        A[5][SLOStatus.UNSATISFIED.value, res_idx, fps_idx, wl_idx] = 0.9

        return A

    def _construct_B_matrix(self):
        """Construct the B matrix (transition model) - Mapping from current states and actions to next states"""
        # Initialize the B matrix with all zeros
        B = utils.obj_array(len(self.state_dims))

        # For each hidden state factor, create a mapping based on actions
        for state_idx in range(len(self.state_dims)):
            B[state_idx] = np.zeros([self.state_dims[state_idx], self.state_dims[state_idx], self.num_actions])

        # B[0]: Transitions for Resolution states based on actions
        # Identity matrix (stay in same state) for all actions except increase/decrease resolution
        for action_idx in range(self.num_actions):
            if action_idx == ActionType.INCREASE_RESOLUTION.value - 1:
                # Increase resolution action
                for i in range(self.num_resolution_states):
                    if i < self.num_resolution_states - 1:
                        B[0][i + 1, i, action_idx] = 0.9  # 90% chance to increase
                        B[0][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[0][i, i, action_idx] = 1.0  # Already at max
            elif action_idx == ActionType.DECREASE_RESOLUTION.value - 1:
                # Decrease resolution action
                for i in range(self.num_resolution_states):
                    if i > 0:
                        B[0][i - 1, i, action_idx] = 0.9  # 90% chance to decrease
                        B[0][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[0][i, i, action_idx] = 1.0  # Already at min
            else:
                # For all other actions, resolution stays the same
                for i in range(self.num_resolution_states):
                    B[0][i, i, action_idx] = 1.0

        # B[1]: Transitions for FPS states based on actions
        for action_idx in range(self.num_actions):
            if action_idx == ActionType.INCREASE_FPS.value - 1:
                # Increase FPS action
                for i in range(self.num_fps_states):
                    if i < self.num_fps_states - 1:
                        B[1][i + 1, i, action_idx] = 0.9  # 90% chance to increase
                        B[1][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[1][i, i, action_idx] = 1.0  # Already at max
            elif action_idx == ActionType.DECREASE_FPS.value - 1:
                # Decrease FPS action
                for i in range(self.num_fps_states):
                    if i > 0:
                        B[1][i - 1, i, action_idx] = 0.9  # 90% chance to decrease
                        B[1][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[1][i, i, action_idx] = 1.0  # Already at min
            else:
                # For all other actions, FPS stays the same
                for i in range(self.num_fps_states):
                    B[1][i, i, action_idx] = 1.0

        # B[2]: Transitions for Work load states based on actions
        for action_idx in range(self.num_actions):
            if action_idx == ActionType.INCREASE_WORK_LOAD.value - 1:
                # Increase work load action
                for i in range(self.num_work_load_states):
                    if i < self.num_work_load_states - 1:
                        B[2][i + 1, i, action_idx] = 0.9  # 90% chance to increase
                        B[2][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[2][i, i, action_idx] = 1.0  # Already at max
            elif action_idx == ActionType.DECREASE_WORK_LOAD.value - 1:
                # Decrease work load action
                for i in range(self.num_work_load_states):
                    if i > 0:
                        B[2][i - 1, i, action_idx] = 0.9  # 90% chance to decrease
                        B[2][i, i, action_idx] = 0.1  # 10% chance to stay the same
                    else:
                        B[2][i, i, action_idx] = 1.0  # Already at min
            else:
                # For all other actions, work load stays the same
                for i in range(self.num_work_load_states):
                    B[2][i, i, action_idx] = 1.0

        return B

    def _construct_C_matrix(self):
        """Construct the C matrix (preference model) - Preferred observations"""
        # Initialize the C matrix with zeros
        C = utils.obj_array(len(self.obs_dims))

        # For each observation modality, set preferences
        for obs_idx in range(len(self.obs_dims)):
            C[obs_idx] = np.zeros(self.obs_dims[obs_idx])

        # Preferences for resolution - higher is better
        # Mild preference for higher resolution
        for i in range(self.num_resolution_states):
            normalized_pref = i / (self.num_resolution_states - 1) * 2.0  # Scale to max of 2.0
            C[0][i] = normalized_pref

        # Preferences for FPS - higher is better
        # Mild preference for higher FPS
        for i in range(self.num_fps_states):
            normalized_pref = i / (self.num_fps_states - 1) * 2.0  # Scale to max of 2.0
            C[1][i] = normalized_pref

        # Preferences for work load - higher is better (better quality)
        # Strong preference for higher work load (higher quality)
        for i in range(self.num_work_load_states):
            normalized_pref = i / (self.num_work_load_states - 1) * 3.0  # Scale to max of 3.0
            C[2][i] = normalized_pref

        # Preferences for queue size - strongly prefer below threshold
        C[3][SLOStatus.SATISFIED.value] = 4.0  # Strong preference for satisfied
        C[3][SLOStatus.UNSATISFIED.value] = -4.0  # Strong aversion to unsatisfied

        # Preferences for FPS SLO - strongly prefer satisfied
        C[4][SLOStatus.SATISFIED.value] = 4.0  # Strong preference for satisfied
        C[4][SLOStatus.UNSATISFIED.value] = -4.0  # Strong aversion to unsatisfied

        # Preferences for resolution SLO - strongly prefer satisfied
        C[5][SLOStatus.SATISFIED.value] = 3.0  # Strong preference for satisfied
        C[5][SLOStatus.UNSATISFIED.value] = -3.0  # Strong aversion to unsatisfied

        return C

    def _construct_D_matrix(self):
        """Construct the D matrix (prior preferences over states) - Initial state beliefs"""
        # Initialize the D matrix with uniform prior over states
        D = utils.obj_array(len(self.state_dims))

        # For each hidden state factor, set initial beliefs based on current state
        for state_idx in range(len(self.state_dims)):
            D[state_idx] = np.ones(self.state_dims[state_idx]) / self.state_dims[state_idx]

        # Set the actual starting state as prior
        D[0][self.elasticity_handler.state_resolution.current_index] = 1.0
        D[1][self.elasticity_handler.state_fps.current_index] = 1.0
        D[2][self.elasticity_handler.state_work_load.current_index] = 1.0

        # Normalize to ensure they are proper probability distributions
        for state_idx in range(len(self.state_dims)):
            D[state_idx] = D[state_idx] / np.sum(D[state_idx])

        return D

    def _get_observations(self) -> list:
        """
        Get current observations of the system

        Returns:
            list: Current observations for all observation modalities
        """
        # TODO: check if work load also needs its own SLO

        # Queue size SLO
        queue_slo_satisfied = self.slo_manager.queue_slo_satisfied()

        # FPS SLO
        fps_slo_satisfied = self.slo_manager.fps_slo_satisfied()

        # Resolution SLO
        resolution_slo_satisfied = self.slo_manager.resolution_slo_satisfied()

        # Construct observation vector
        observations = [
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index,
            SLOStatus.SATISFIED.value if queue_slo_satisfied else SLOStatus.UNSATISFIED.value,
            SLOStatus.SATISFIED.value if fps_slo_satisfied else SLOStatus.UNSATISFIED.value,
            SLOStatus.SATISFIED.value if resolution_slo_satisfied else SLOStatus.UNSATISFIED.value
        ]

        return observations

    def _perform_action(self, action_idx: int) -> bool:
        """
        Perform the selected action using match-case structure.

        Args:
            action_idx: Index of the action to perform

        Returns:
            bool: True if the action was successful, False otherwise
        """
        action = self.actions[action_idx]

        match action:
            case ActionType.DO_NOTHING:
                return True  # Do nothing is always successful
            case ActionType.INCREASE_RESOLUTION:
                return self.elasticity_handler.increase_resolution()
            case ActionType.DECREASE_RESOLUTION:
                return self.elasticity_handler.decrease_resolution()
            case ActionType.INCREASE_FPS:
                return self.elasticity_handler.increase_fps()
            case ActionType.DECREASE_FPS:
                return self.elasticity_handler.decrease_fps()
            case ActionType.INCREASE_WORK_LOAD:
                return self.elasticity_handler.increase_work_load()
            case ActionType.DECREASE_WORK_LOAD:
                return self.elasticity_handler.decrease_work_load()
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the active inference loop

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        # Get current observations
        observations = self._get_observations()

        # Perform active inference
        q_pi, q_s = self.agent.infer_states(observations)
        action = self.agent.sample_action()

        # Perform the selected action
        success = self._perform_action(action)

        # Update agent's beliefs based on the action taken
        self.agent.infer_states(observations, action)

        return self.actions[action], success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()


