import logging
import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.view.aif_agent_observations import AIFAgentObservations
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

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
    OBS_INFERENCE_QUALITY_INDEX = 2
    OBS_QUEUE_SIZE_INDEX = 3
    OBS_MEMORY_USAGE_INDEX = 4
    OBS_GLOBAL_PROCESSING_TIME_INDEX = 5
    OBS_WORKER_PROCESSING_TIME_INDEX = 6

    # Action indices
    ACTION_RESOLUTION_INDEX = 0
    ACTION_FPS_INDEX = 1
    ACTION_INFERENCE_QUALITY_INDEX = 2

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


    def __init__(self, elasticity_handler: ElasticityHandler, request_handler: RequestHandler, task_generator: TaskGenerator, policy_length: int = 1, track_slo_stats=True):
        """
        Initialize the Active Inference Agent

        Args:
            elasticity_handler: The handler for changing system parameters
            policy_length: Number of time steps to plan ahead (default: 1)
        """
        super().__init__(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)

        self.observations = AIFAgentObservations(elasticity_handler.observations(),self._slo_manager)

        # Get the relative actions view for clean interface to increase/decrease actions
        self.relative_actions = elasticity_handler.actions_relative()

        self.num_resolution_states = len(elasticity_handler.state_resolution.possible_states)
        self.num_fps_states = len(elasticity_handler.state_fps.possible_states)
        self.num_inference_quality_states = len(elasticity_handler.state_inference_quality.possible_states)

        # Define actions (for each state dimension)
        self.resolution_actions = list(ActionType)
        self.fps_actions = list(ActionType)
        self.work_load_actions = list(ActionType)

        self.num_resolution_actions = len(self.resolution_actions)
        self.num_fps_actions = len(self.fps_actions)
        self.num_work_load_actions = len(self.work_load_actions)

        self.control_dims = [
            self.num_resolution_actions,
            self.num_fps_actions,
            self.num_work_load_actions
        ]

        # 3 states for SLO status: SATISFIED, WARNING, UNSATISFIED
        self.num_queue_states = len(SloStatus)
        self.num_memory_states = len(SloStatus)
        self.num_global_processing_time_states = len(SloStatus)
        self.num_worker_processing_time_states = len(SloStatus)

        self.num_slo = 4

        # policy settings
        self.policy_length = policy_length

        self._setup_generative_model()

    def step(self):
        """
        Perform a single step of the active inference loop

        Returns:
            tuple[GeneralActionType, bool]: The action taken and whether it was successful
        """
        observations = self._get_observations()

        # Perform active inference, q_s = Q(s) = Posterior believes Q over hidden states  s
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()

        actions_idx = np.array(self.agent.sample_action(), dtype=int).tolist()
        print(f'sampled actions: {actions_idx}')

        success = self._perform_actions(actions_idx)
        return ActionType.NONE, success


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
            self.num_inference_quality_states,  # Work load state
            self.num_queue_states,  # Queue size status (OK, WARNING, CRITICAL)
            self.num_memory_states,  # Memory usage status (OK, WARNING, CRITICAL)
            self.num_global_processing_time_states,  # Global processing time status (OK, WARNING, CRITICAL)
            self.num_worker_processing_time_states,  # Worker processing time status (OK, WARNING, CRITICAL)
        ]

        # Define state dimensions (hidden states)
        self.state_dims = [
            self.num_resolution_states,  # Resolution state
            self.num_fps_states,  # FPS state
            self.num_inference_quality_states,  # Work load state
        ]

        # In pymdp, there's a fundamental assumption that each state dimension can be controlled by a
        # separate action dimension. This has 3 state dimensions (Resolution, fps, workload), so
        # pymdp agent.sample_action() function is returning an action for each dimension - hence the array of 3 values.

        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        #self.agent = Agent(A=A, B=B, C=C, D=D)
        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            num_controls=self.control_dims,
            policy_len=self.policy_length,
            control_fac_idx=[self.OBS_RESOLUTION_INDEX, self.OBS_FPS_INDEX, self.OBS_INFERENCE_QUALITY_INDEX],  # hidden state factors that are directly controllable
            inference_algo="VANILLA",
            action_selection="deterministic"
        )

    def _construct_A_matrix(self):
        """
        Construct the A matrix (observation model) - Likelihood mapping from hidden states to observations

        The A array encodes the likelihood mapping between hidden states and observations.
        The A array answers: "Given a hidden state, what observation am I likely to see?"

        A[observation][slo-status, resolution-state, fps-state, workload-state] = probability (0 - 1)
        """
        # TODO Idea. replace slo-status with dimensionality for each slo status, i.e. remove observation 3 & 4 and embedd them
        
        #  A[observation][slo-status_qsize, slo_status_mem, resolution-state, fps-state, workload-state] = probability (0 - 1)
        A = utils.obj_array(len(self.obs_dims))

        # Initialize observation modalities
        for obs_idx, obs_dim in enumerate(self.obs_dims):
            A[obs_idx] = np.zeros((obs_dim, *self.state_dims))

        # Observation 0: Resolution state - direct mapping
        for i in range(self.num_resolution_states):
            A[self.OBS_RESOLUTION_INDEX][i, i, :, :] = 1.0

        # Observation 1: FPS state - direct mapping
        for i in range(self.num_fps_states):
            A[self.OBS_FPS_INDEX][i, :, i, :] = 1.0

        # Observation 2: Work load state - direct mapping
        for i in range(self.num_inference_quality_states):
            A[self.OBS_INFERENCE_QUALITY_INDEX][i, :, :, i] = 1.0

        # SLO probability mappings: All 4 SLO status observations
        queue_probs, mem_probs, global_processing_probs, worker_processing_probs = self.observations.get_all_slo_probabilities()
        
        for res in range(self.num_resolution_states):
            for fps in range(self.num_fps_states):
                for wl in range(self.num_inference_quality_states):
                    A[self.OBS_QUEUE_SIZE_INDEX][:, res, fps, wl] = queue_probs
                    A[self.OBS_MEMORY_USAGE_INDEX][:, res, fps, wl] = mem_probs
                    A[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][:, res, fps, wl] = global_processing_probs
                    A[self.OBS_WORKER_PROCESSING_TIME_INDEX][:, res, fps, wl] = worker_processing_probs

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

        probability_to_change_state = 1.0

        # State 0 - Resolution: Transitions for Resolution states based on actions
        B[self.OBS_RESOLUTION_INDEX] = self._construct_sub_transition_model(self.num_resolution_states,
                                                                                            self.num_resolution_actions,
                                                                                            probability_to_change_state)

        # State 1 - FPS: Transitions for FPS states based on actions
        B[self.OBS_FPS_INDEX] = self._construct_sub_transition_model(self.num_fps_states,
                                                                                     self.num_fps_actions,
                                                                                     probability_to_change_state)

        # State 2 - Work Load: Transitions for Work load states based on actions
        B[self.OBS_INFERENCE_QUALITY_INDEX] = self._construct_sub_transition_model(self.num_inference_quality_states,
                                                                                                   self.num_work_load_actions,
                                                                                                   probability_to_change_state)

        return B

    def _construct_sub_transition_model(self, num_states: int, num_actions: int, probability_to_change_state: float):
        """
        Construct a  sub-B matrix (transition model) - Mapping from current states and actions to next states
        Specifies the probability of moving from one hidden state to another, given a particular action.
        B[s', s, a] = probability of transitioning from state s to s' under action a
        B[s', s, a] = (0-1)
        """
        B = np.zeros((num_states, num_states, num_actions))

        for action in range(num_actions):
            for state in range(num_states):
                if action == ActionType.INCREASE.value:
                    # increasing value leaves you at state+1 or at max if you cannot increase further
                    next_state = min(state + 1, num_states - 1)
                elif action == ActionType.DECREASE.value:
                    # decreasing value leaves you at state-11 or at 0 if you cannot decrease further
                    next_state = max(state - 1, 0)
                else:
                    # no action = no change in state
                    next_state = state

                B[next_state, state, action] = probability_to_change_state

        return B

    def _construct_C_matrix(self):
        """
        Construct the C matrix (preference model) - Preferred observations
        C[observation-type, observation] = reward
        """
        C = utils.obj_array(len(self.obs_dims))

        # For each observation modality, set preferences
        for obs_idx, obs_dim in enumerate(self.obs_dims):
            C[obs_idx] = np.zeros(obs_dim)

        # Preferences for resolution - higher is better (linear scaling)
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        max_res = self.num_resolution_states - 1
        C[self.OBS_RESOLUTION_INDEX][:] = [self.MEDIUM_PREFERENCE * (i / max_res) for i in
                                           range(self.num_resolution_states)]

        # Preferences for FPS - higher is better (linear scaling)
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        max_fps = self.num_fps_states - 1
        C[self.OBS_FPS_INDEX][:] = [self.MEDIUM_PREFERENCE * (i / max_fps) for i in range(self.num_fps_states)]

        # Preferences for inference quality - higher is better (linear scaling)
        # Each level has a mildly higher preference then the one before, Scale to max defined via PREFERENCE
        max_inference_quality = self.num_inference_quality_states - 1
        C[self.OBS_INFERENCE_QUALITY_INDEX][:] = [self.LOW_PREFERENCE * (i / max_inference_quality) for i in
                                                  range(self.num_inference_quality_states)]

        # Preferences for queue size
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value] = self.STRONG_PREFERENCE
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value] = self.STRONG_AVERSION

        # Preferences for memory usage
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value] = self.STRONG_PREFERENCE
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value] = self.STRONG_AVERSION

        # Preferences for global processing time
        C[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][SloStatus.OK.value] = self.STRONG_PREFERENCE
        C[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][SloStatus.CRITICAL.value] = self.STRONG_AVERSION

        # Preferences for worker processing time
        C[self.OBS_WORKER_PROCESSING_TIME_INDEX][SloStatus.OK.value] = self.STRONG_PREFERENCE
        C[self.OBS_WORKER_PROCESSING_TIME_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_WORKER_PROCESSING_TIME_INDEX][SloStatus.CRITICAL.value] = self.STRONG_AVERSION

        return C

    def _construct_D_matrix(self):
        """Construct the D matrix (prior believes over states) - Initial beliefs, i.e. what states are expected before making an observation"""
        D = utils.obj_array(len(self.state_dims))

        current_states = [
            self.observations._elasticity_observations.get_current_resolution_state().current_index,
            self.observations._elasticity_observations.get_current_fps_state().current_index,
            self.observations._elasticity_observations.get_current_inference_quality_state().current_index
        ]

        for i, state in enumerate(current_states):
            D[i] = np.zeros(self.state_dims[i])
            D[i][state] = 1.0

        return D

    def _get_observations(self) -> list[int]:
        """
        Get current observations of the system

        Returns:
            list: Current observations for all observation modalities
        """
        return self.observations.get_observations()

    def _perform_action(self, action: GeneralActionType) -> bool:
        """
        Perform the selected action using the relative actions view for increase/decrease operations.

        Args:
            action: The action to perform

        Returns:
            bool: True if the action was successful, False otherwise
        """

        match action:
            case GeneralActionType.NONE:
                return True
            case GeneralActionType.INCREASE_RESOLUTION:
                return self.relative_actions.increase_resolution()
            case GeneralActionType.DECREASE_RESOLUTION:
                return self.relative_actions.decrease_resolution()
            case GeneralActionType.INCREASE_FPS:
                return self.relative_actions.increase_fps()
            case GeneralActionType.DECREASE_FPS:
                return self.relative_actions.decrease_fps()
            case GeneralActionType.INCREASE_INFERENCE_QUALITY:
                return self.relative_actions.increase_inference_quality()
            case GeneralActionType.DECREASE_INFERENCE_QUALITY:
                return self.relative_actions.decrease_inference_quality()
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def _perform_actions(self, actions: list[int]):
        """Tries to perform for all state dimensions using the relative actions view"""
        # TODO: Return list of actions and if they were successful
        success = True

        # Resolution action
        if actions[self.ACTION_RESOLUTION_INDEX] == ActionType.INCREASE:
            success &= self.relative_actions.increase_resolution()
        elif actions[self.ACTION_RESOLUTION_INDEX] == ActionType.DECREASE:
            success &= self.relative_actions.decrease_resolution()

        # FPS action
        if actions[self.ACTION_FPS_INDEX] == ActionType.INCREASE:
            success &= self.relative_actions.increase_fps()
        elif actions[self.ACTION_FPS_INDEX] == ActionType.DECREASE:
            success &= self.relative_actions.decrease_fps()

        # Workload action
        if actions[self.ACTION_INFERENCE_QUALITY_INDEX] == ActionType.INCREASE:
            success &= self.relative_actions.increase_inference_quality()
        elif actions[self.ACTION_INFERENCE_QUALITY_INDEX] == ActionType.DECREASE:
            success &= self.relative_actions.decrease_inference_quality()

        return success