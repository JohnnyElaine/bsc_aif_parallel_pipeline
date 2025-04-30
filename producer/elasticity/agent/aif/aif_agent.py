import logging
import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus

log = logging.getLogger('producer')


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
    VERY_STRONG_PREFERENCE = 4.0
    STRONG_PREFERENCE = VERY_STRONG_PREFERENCE * 0.75
    MEDIUM_PREFERENCE = VERY_STRONG_PREFERENCE * 0.5
    LOW_PREFERENCE = VERY_STRONG_PREFERENCE * 0.25
    VERY_LOW_PREFERENCE = VERY_STRONG_PREFERENCE * 0.1
    NEUTRAL = 0.0
    VERY_LOW_AVERSION = -VERY_LOW_PREFERENCE
    LOW_AVERSION = -LOW_PREFERENCE
    MEDIUM_AVERSION = -MEDIUM_PREFERENCE
    STRONG_AVERSION = -STRONG_PREFERENCE
    VERY_STRONG_AVERSION = -VERY_STRONG_PREFERENCE


    def __init__(self, elasticity_handler: ElasticityHandler, policy_length: int = 1):
        """
        Initialize the Active Inference Agent

        Args:
            elasticity_handler: The handler for changing system parameters
            policy_length: Number of time steps to plan ahead (default: 2)
        """
        super().__init__(elasticity_handler)

        possible_resolutions = elasticity_handler.state_resolution.possible_states
        possible_fps = elasticity_handler.state_fps.possible_states
        possible_work_loads = elasticity_handler.state_work_load.possible_states

        # Define the dimensions of various observations
        self.num_resolution_states = len(possible_resolutions)
        self.num_fps_states = len(possible_fps)
        self.num_work_load_states = len(possible_work_loads)

        # Define actions (for each state dimension)
        self.resolution_actions = list(ActionType)
        self.fps_actions = list(ActionType)
        self.work_load_actions = list(ActionType)

        self.control_dims = [len(self.resolution_actions), len(self.fps_actions), len(self.work_load_actions)]

        # 3 states for SLO status: SATISFIED, WARNING, UNSATISFIED
        self.num_queue_states = len(SloStatus)
        self.num_memory_states = len(SloStatus)

        self.num_slo = 2

        # Define all actions
        self.actions = list(GeneralActionType)
        self.num_actions = len(self.actions)

        # policy settings
        self.policy_length = policy_length

        self._setup_generative_model()

    def step(self) -> tuple[GeneralActionType, bool]:
        """
        Perform a single step of the active inference loop

        Returns:
            tuple[GeneralActionType, bool]: The action taken and whether it was successful
        """
        observations = self._get_observations()

        # Perform active inference, q_s = Q(s) = Posterior believes Q over hidden states  s
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()

        actions = np.array(self.agent.sample_action(), dtype=int).tolist()
        print(f'sampled actions: {actions}')

        #self._perform_actions(actions)
        #return ActionType.NONE, True

        are_slos_satisfied = all(slo == SloStatus.OK for slo in observations[-self.num_slo:])
        action_to_perform = self._select_action(actions, are_slos_satisfied)
        # Perform the selected action
        success = self._perform_action(action_to_perform)

        # Update agent's beliefs based on the action taken
        # Note: should not be needed, because we get the observations dynamically
        #self.agent.infer_states(observations, action)

        return action_to_perform, success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()

    def _setup_generative_model(self):
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

        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        self.agent = Agent(A=A, B=B, C=C, D=D)

    def _construct_A_matrix(self):
        """
        Construct the A matrix (observation model) - Likelihood mapping from hidden states to observations

        The A array encodes the likelihood mapping between hidden states and observations.
        The A array answers: "Given a hidden state, what observation am I likely to see?"

        A[observation][slo-status, resolution-state, fps-state, workload-state] = probability (0 - 1)
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
                    queue_probs = self.slo_manager.get_qsize_slo_state_probabilities()

                    # Set probabilities for each SLO state
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.OK.value]
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.WARNING.value]
                    A[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value, res_idx, fps_idx, wl_idx] = queue_probs[SloStatus.CRITICAL.value]

        # Observation 4: Memory usage status - depends on resolution, FPS, and work load
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    # Get probability distributions for memory SLO states
                    memory_probs = self.slo_manager.get_mem_slo_state_probabilities()

                    # Set probabilities for each SLO state
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.OK.value]
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.WARNING.value]
                    A[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value, res_idx, fps_idx, wl_idx] = memory_probs[SloStatus.CRITICAL.value]
        return A

    def _construct_B_matrix(self):
        """
        Construct the B matrix (transition model) - Mapping from current states and actions to next states.
        Specifies the probability of moving from one hidden state to another, given a particular action.
        B[s', s, a] = probability of transitioning from state s to s' under action a
        B[s', s, a] = (0-1)

        Because have multiple hidden state factors, the B becomes a list of arrays, one for each hidden state factor
        B[state-factor][s', s, a]
        """
        B = utils.obj_array(len(self.state_dims))

        # TODO: check if probability_to_change_state = 1 (deterministic agent) is preffered
        probability_to_change_state = 1

        # State 0 - Resolution: Transitions for Resolution states based on actions
        B[ActiveInferenceAgent.OBS_RESOLUTION_INDEX] = self._construct_sub_transition_model(self.num_resolution_states,
                                                                                            self.num_actions,
                                                                                            GeneralActionType.INCREASE_RESOLUTION,
                                                                                            GeneralActionType.DECREASE_RESOLUTION,
                                                                                            probability_to_change_state)

        # State 1 - FPS: Transitions for FPS states based on actions
        B[ActiveInferenceAgent.OBS_FPS_INDEX] = self._construct_sub_transition_model(self.num_fps_states, self.num_actions,
                                                                                     GeneralActionType.INCREASE_FPS,
                                                                                     GeneralActionType.DECREASE_FPS,
                                                                                     probability_to_change_state)

        # State 2 - Work Load: Transitions for Work load states based on actions
        B[ActiveInferenceAgent.OBS_WORK_LOAD_INDEX] = self._construct_sub_transition_model(self.num_work_load_states, self.num_actions,
                                                                                     GeneralActionType.INCREASE_WORK_LOAD,
                                                                                     GeneralActionType.DECREASE_WORK_LOAD,
                                                                                     probability_to_change_state)

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
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        for i in range(self.num_resolution_states):
            normalized_pref = i / max(self.num_resolution_states - 1, 1) * ActiveInferenceAgent.LOW_PREFERENCE
            C[ActiveInferenceAgent.OBS_RESOLUTION_INDEX][i] = normalized_pref

        # Preferences for FPS - higher is better
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        for i in range(self.num_fps_states):
            normalized_pref = i / max(self.num_fps_states - 1, 1) * ActiveInferenceAgent.LOW_PREFERENCE
            C[ActiveInferenceAgent.OBS_FPS_INDEX][i] = normalized_pref

        # Preferences for work load - higher is better (better quality)
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        for i in range(self.num_work_load_states):
            normalized_pref = i / max(self.num_work_load_states - 1, 1) * ActiveInferenceAgent.VERY_LOW_PREFERENCE # less prio for increasing work load
            C[ActiveInferenceAgent.OBS_WORK_LOAD_INDEX][i] = normalized_pref

        # Preferences for queue size
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value] = ActiveInferenceAgent.VERY_STRONG_PREFERENCE
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value] = ActiveInferenceAgent.NEUTRAL
        C[ActiveInferenceAgent.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value] = ActiveInferenceAgent.VERY_STRONG_AVERSION

        # Preferences for memory usage
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value] = ActiveInferenceAgent.VERY_STRONG_PREFERENCE
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value] = ActiveInferenceAgent.NEUTRAL
        C[ActiveInferenceAgent.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value] = ActiveInferenceAgent.VERY_STRONG_AVERSION

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

    def _construct_sub_transition_model(self, num_states: int, num_actions: int, increase_action: GeneralActionType,
                                        decrease_action: GeneralActionType, probability_to_change_state: float):
        """
        Construct a  sub-B matrix (transition model) - Mapping from current states and actions to next states
        Specifies the probability of moving from one hidden state to another, given a particular action.
        B[s', s, a] = probability of transitioning from state s to s' under action a
        B[s', s, a] = (0-1)
        """

        B = np.zeros((num_states, num_states, num_actions))

        # Identity matrix (stay in same state) for all actions except increase/decrease
        for action_idx in range(num_actions):
            if action_idx == increase_action.value:
                for i in range(num_states):
                    if i == num_states - 1:  # Maximum state - can't increase further
                        B[i, i, action_idx] = 1.0
                    else:
                        B[i + 1, i, action_idx] = probability_to_change_state
                        B[i, i, action_idx] = 1 - probability_to_change_state

            elif action_idx == decrease_action.value:
                for i in range(num_states):
                    if i == 0:  # Minimum state - can't decrease further
                        B[i, i, action_idx] = 1.0
                    else:
                        B[i - 1, i, action_idx] = probability_to_change_state
                        B[i, i, action_idx] = 1 - probability_to_change_state
            else:
                # For all other actions, state stays the same
                for i in range(num_states):
                    B[i, i, action_idx] = 1.0

        return B

    def _get_observations(self) -> list[int]:
        """
        Get current observations of the system

        Returns:
            list: Current observations for all observation modalities
        """

        queue_slo_status, memory_slo_status = self.slo_manager.get_all_slo_status(track_stats=True)

        observations = [
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index,
            queue_slo_status.value,
            memory_slo_status.value
        ]

        return observations

    def _perform_action(self, action: GeneralActionType) -> bool:
        """
        Perform the selected action using match-case structure.

        Args:
            action: The action to perform

        Returns:
            bool: True if the action was successful, False otherwise
        """

        match action:
            case GeneralActionType.NONE:
                return True
            case GeneralActionType.INCREASE_RESOLUTION:
                return self.elasticity_handler.increase_resolution()
            case GeneralActionType.DECREASE_RESOLUTION:
                return self.elasticity_handler.decrease_resolution()
            case GeneralActionType.INCREASE_FPS:
                return self.elasticity_handler.increase_fps()
            case GeneralActionType.DECREASE_FPS:
                return self.elasticity_handler.decrease_fps()
            case GeneralActionType.INCREASE_WORK_LOAD:
                return self.elasticity_handler.increase_work_load()
            case GeneralActionType.DECREASE_WORK_LOAD:
                return self.elasticity_handler.decrease_work_load()
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def _perform_actions(self, actions_idxs: list):
        """Tries to perform all actions given by actions_idxs"""
        for action_idx in actions_idxs:
            self._perform_action(self.actions[action_idx])

    def _select_action(self, actions_idxs: list[int], slo_satisfied) -> GeneralActionType:
        """Smart action selection that considers SLO status"""

        # If SLOs are violated, prioritize corrective actions
        if not slo_satisfied:
            for action_idx in actions_idxs:
                action = self.actions[action_idx]
                if action in [GeneralActionType.DECREASE_RESOLUTION,
                              GeneralActionType.DECREASE_FPS,
                              GeneralActionType.DECREASE_WORK_LOAD]:
                    return action

        # Otherwise use normal preference-driven selection
        for action_idx in actions_idxs:
            action = self.actions[int(action_idx)]
            if action != GeneralActionType.NONE:
                return action

        return GeneralActionType.NONE