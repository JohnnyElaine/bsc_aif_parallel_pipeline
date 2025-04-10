import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus


class ActiveInferenceAgent(ElasticityAgent):
    """
    Active Inference Agent that uses the Free Energy Principle to maintain
    Service Level Objectives (SLOs) in a distributed video processing system.

    The agent monitors the system's state and takes actions to optimize the quality
    of experience while ensuring computational resources are properly utilized.
    """
    # Observations
    OBS_RESOLUTION_INDEX = 0
    OBS_FPS_INDEX = 1
    OBS_WORK_LOAD_INDEX = 2
    OBS_QUEUE_SIZE_INDEX = 3
    OBS_MEMORY_USAGE_INDEX = 4

    # Observation preferences
    STRONG_PREFERENCE = 4.0
    MEDIUM_PREFERENCE = 2.0
    LOW_PREFERENCE = 1.0
    VERY_LOW_PREFERENCE = 0.5
    NEUTRAL = 0.0
    VERY_LOW_AVERSION = -VERY_LOW_PREFERENCE
    LOW_AVERSION = -LOW_PREFERENCE
    MEDIUM_AVERSION = -MEDIUM_PREFERENCE
    STRONG_AVERSION = -STRONG_PREFERENCE


    def __init__(self, elasticity_handler: ElasticityHandler, planning_horizon: int = 2):
        """
        Initialize the Active Inference Agent

        Args:
            elasticity_handler: The handler for changing system parameters
            planning_horizon: Number of time steps to plan ahead (default: 2)
        """
        super().__init__(elasticity_handler)
        self.possible_resolutions = elasticity_handler.state_resolution.possible_states
        self.possible_fps = elasticity_handler.state_fps.possible_states
        self.possible_work_loads = elasticity_handler.state_work_load.possible_states

        # Define the dimensions of various observations
        self.num_resolution_states = len(self.possible_resolutions)
        self.num_fps_states = len(self.possible_fps)
        self.num_work_load_states = len(self.possible_work_loads)

        # 3 states for SLO status: SATISFIED, WARNING, UNSATISFIED
        self.num_queue_states = len(SloStatus)
        self.num_memory_states = len(SloStatus)

        # Define actions
        self.actions = list(ActionType)
        self.num_actions = len(self.actions)

        self._setup_generative_model(planning_horizon)

    def step(self) -> tuple[ActionType, bool]:
        """
        Perform a single step of the active inference loop

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        # Get current observations
        observations = self._get_observations()

        # Perform active inference, q_s = Q(s) = Posterior believes Q over hidden states  s
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()
        print(f'sampled action: {action}') # TODO check which actions are returned

        resolution_action = int(action[0])  # First element controls resolution
        fps_action = int(action[1])  # Second element controls FPS
        work_load_action = int(action[2])  # Third element controls work load

        action_to_perform = resolution_action

        # Perform the selected action
        success = self._perform_action(action_to_perform)

        # Update agent's beliefs based on the action taken
        # Note: should not be needed, because we get the observations dynamically
        #self.agent.infer_states(observations, action)

        return self.actions[action_to_perform], success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()

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
            self.num_memory_states,  # Memory usage status (OK or too high)
        ]

        # Define state dimensions (hidden states)
        self.state_dims = [
            self.num_resolution_states,  # Resolution state
            self.num_fps_states,  # FPS state
            self.num_work_load_states,  # Work load state
        ]
        # In pymdp, there's a fundamental assumption that each state dimension can be controlled by a
        # separate action dimension. This has 3 state dimensions (Resolution, fps, workload), so
        # pymdp agent.sample_action() function is returning an action for each dimension - hence the array of 3 values.


        self.control_dims = [self.num_actions]

        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        self.agent = Agent(A=A, B=B, C=C, D=D)

    def _construct_A_matrix(self):
        """
        Construct the A matrix (observation model) - Likelihood mapping from hidden states to observations
        A[observation, slo-status, resolution-state, fps-state, workload-state] = probability (0 - 1)
        """
        A = utils.obj_array(len(self.obs_dims))

        # For each observation modality, create a mapping from hidden states
        for obs_idx in range(len(self.obs_dims)):
            A[obs_idx] = np.zeros(self.obs_dims[obs_idx:obs_idx + 1] + self.state_dims)

        # Observation 0: Resolution state - direct mapping
        for i in range(self.num_resolution_states):
            A[ActiveInferenceAgent.OBS_RESOLUTION_INDEX][i, i, :, :] = 1.0

        # Observation 1: FPS state - direct mapping
        for i in range(self.num_fps_states):
            A[ActiveInferenceAgent.OBS_FPS_INDEX][i, :, i, :] = 1.0

        # Observation 2: Work load state - direct mapping
        for i in range(self.num_work_load_states):
            A[ActiveInferenceAgent.OBS_WORK_LOAD_INDEX][i, :, :, i] = 1.0

        # Observation 3: Queue size status - depends on FPS and Work load
        # Higher FPS and higher work load increase probability of large queue
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    # Get probability distributions for queue SLO states
                    queue_probs = self.slo_manager.get_queue_slo_state_probabilities()

                    # Set probabilities for each SLO state
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.OK.value]
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.WARNING.value]
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.CRITICAL.value]

        # Observation 4: Memory usage status - depends on resolution, FPS, and work load
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    # Get probability distributions for memory SLO states
                    memory_probs = self.slo_manager.get_memory_slo_state_probabilities()

                    # Set probabilities for each SLO state
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.OK.value]
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.WARNING.value]
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.CRITICAL.value]
        return A

    def _construct_B_matrix(self):
        """
        Construct the B matrix (transition model) - Mapping from current states and actions to next states
        B[current-state, next-state, action] = probability (0 - 1)
        """
        B = utils.obj_array(len(self.state_dims))

        # For each hidden state factor, create a mapping based on actions
        for state_idx in range(len(self.state_dims)):
            B[state_idx] = np.zeros([self.state_dims[state_idx], self.state_dims[state_idx], self.num_actions])

        # TODO: check if i should change probability to 100% & 0% to guarantee change of state
        probability_to_change_state = 0.9

        # State 0 - Resolution: Transitions for Resolution states based on actions
        self._construct_sub_transition_model(B, ActiveInferenceAgent.OBS_RESOLUTION_INDEX, self.num_actions,
                                             self.num_resolution_states, ActionType.INCREASE_RESOLUTION,
                                             ActionType.DECREASE_RESOLUTION, probability_to_change_state)

        # State 1 - FPS: Transitions for FPS states based on actions
        self._construct_sub_transition_model(B, ActiveInferenceAgent.OBS_FPS_INDEX, self.num_actions,
                                             self.num_fps_states, ActionType.INCREASE_FPS,
                                             ActionType.DECREASE_FPS, probability_to_change_state)

        # State 2 - Work Load: Transitions for Work load states based on actions
        self._construct_sub_transition_model(B, ActiveInferenceAgent.OBS_WORK_LOAD_INDEX, self.num_actions,
                                             self.num_work_load_states, ActionType.INCREASE_WORK_LOAD,
                                             ActionType.DECREASE_WORK_LOAD, probability_to_change_state)

        return B

    def _construct_C_matrix(self):
        """
        Construct the C matrix (preference model) - Preferred observations
        C[observation-type, observation] = reward
        """
        C = utils.obj_array(len(self.obs_dims))

        # For each observation modality, set preferences
        for obs_idx in range(len(self.obs_dims)):
            C[obs_idx] = np.zeros(self.obs_dims[obs_idx])

        # Preferences for resolution - higher is better
        # Mild preference for higher resolution
        for i in range(self.num_resolution_states):
            normalized_pref = i / max(self.num_resolution_states - 1, 1) * ActiveInferenceAgent.LOW_PREFERENCE # Scale to max of 2.0
            C[ActiveInferenceAgent.OBS_RESOLUTION_INDEX][i] = normalized_pref

        # Preferences for FPS - higher is better
        # Mild preference for higher FPS
        for i in range(self.num_fps_states):
            normalized_pref = i / max(self.num_fps_states - 1, 1) * ActiveInferenceAgent.LOW_PREFERENCE   # Scale to max of 2.0
            C[ActiveInferenceAgent.OBS_FPS_INDEX][i] = normalized_pref

        # Preferences for work load - higher is better (better quality)
        # Mild preference for higher work load (higher quality)
        for i in range(self.num_work_load_states):
            normalized_pref = i / max(self.num_work_load_states - 1, 1) * ActiveInferenceAgent.LOW_PREFERENCE  # Scale to max of 2.0
            C[ActiveInferenceAgent.OBS_WORK_LOAD_INDEX][i] = normalized_pref

        # Preferences for queue size - 3 states now
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value] = ActiveInferenceAgent.STRONG_PREFERENCE
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value] = ActiveInferenceAgent.NEUTRAL
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value] = ActiveInferenceAgent.STRONG_AVERSION

        # Preferences for memory usage - 3 states now
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value] = ActiveInferenceAgent.STRONG_PREFERENCE
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value] = ActiveInferenceAgent.NEUTRAL
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value] = ActiveInferenceAgent.STRONG_AVERSION

        return C

    def _construct_D_matrix(self):
        """Construct the D matrix (prior preferences over states) - Initial state beliefs"""
        # Initialize the D matrix with uniform prior over states
        D = utils.obj_array(len(self.state_dims))

        # For each hidden state factor, set initial beliefs based on current state
        for state_idx in range(len(self.state_dims)):
            D[state_idx] = np.ones(self.state_dims[state_idx]) / self.state_dims[state_idx]

        # Here's the problematic part - you're trying to set specific indices to 1.0
        # which breaks the probability distribution. Instead:

        # Reset to zeros first
        for state_idx in range(len(self.state_dims)):
            D[state_idx] = np.zeros(self.state_dims[state_idx])

        # Set the actual starting state as prior (with probability 1.0)
        # State 0 - Resolution
        D[ActiveInferenceAgent.OBS_RESOLUTION_INDEX][self.elasticity_handler.state_resolution.current_index] = 1.0
        # State 1 - FPS
        D[ActiveInferenceAgent.OBS_FPS_INDEX][self.elasticity_handler.state_fps.current_index] = 1.0
        # State 2 - Work Load
        D[ActiveInferenceAgent.OBS_WORK_LOAD_INDEX][self.elasticity_handler.state_work_load.current_index] = 1.0

        # Ensure they are proper probability distributions
        for state_idx in range(len(self.state_dims)):
            if np.sum(D[state_idx]) == 0:  # Handle case where index might be out of bounds
                D[state_idx] = np.ones(self.state_dims[state_idx]) / self.state_dims[state_idx]
            else:
                D[state_idx] = D[state_idx] / np.sum(D[state_idx])

        return D

    def _construct_sub_transition_model(self,
                                        B: np.ndarray,
                                        state_index: int,
                                        num_actions: int,
                                        num_states: int,
                                        increase_action: ActionType,
                                        decrease_action: ActionType,
                                        probability_to_change_state: float):
        """
        Construct a  sub-B matrix (transition model) - Mapping from current states and actions to next states
        B[current-state, next-state, action] = probability (0 - 1)
        """
        # Identity matrix (stay in same state) for all actions except increase/decrease
        for action_idx in range(num_actions):
            if action_idx == decrease_action.value: # TODO: maybe switch with increase_action.value
                for i in range(num_states):
                    if i < num_states - 1:
                        B[state_index][i + 1, i, action_idx] = probability_to_change_state
                        B[state_index][i, i, action_idx] = 1 - probability_to_change_state
                    else:
                        B[state_index][i, i, action_idx] = 1.0  # Already at max
            elif action_idx == increase_action.value:
                for i in range(num_states): # TODO: maybe switch with decrease_action.value
                    if i > 0:
                        B[state_index][i - 1, i, action_idx] = probability_to_change_state
                        B[state_index][i, i, action_idx] = 1 - probability_to_change_state
                    else:
                        B[state_index][i, i, action_idx] = 1.0  # Already at min
            else:
                # For all other actions, state stays the same
                for i in range(num_states):
                    B[state_index][i, i, action_idx] = 1.0

    def _get_observations(self) -> list[int]:
        """
        Get current observations of the system

        Returns:
            list: Current observations for all observation modalities
        """
        queue_slo_status = self.slo_manager.get_qsize_slo_status().value
        memory_slo_status = self.slo_manager.get_memory_slo_status().value

        observations = [
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index,
            queue_slo_status,
            memory_slo_status
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
                return True
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