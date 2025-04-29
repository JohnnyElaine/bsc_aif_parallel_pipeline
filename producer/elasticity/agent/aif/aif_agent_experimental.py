import logging
import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus

log = logging.getLogger('producer')


class ActiveInferenceAgentExperimental(ElasticityAgent):
    """
    Active Inference Agent that uses the Free Energy Principle to maintain
    Service Level Objectives (SLOs) in a distributed video processing system.
    """
    # Observations
    OBS_RESOLUTION_INDEX = 0
    OBS_FPS_INDEX = 1
    OBS_WORK_LOAD_INDEX = 2
    OBS_QUEUE_SIZE_INDEX = 3
    OBS_MEMORY_USAGE_INDEX = 4

    # ACTIONS
    ACTION_RESOLUTION_INDEX = OBS_RESOLUTION_INDEX
    ACTION_FPS_INDEX = OBS_FPS_INDEX
    ACTION_WORK_LOAD_INDEX = OBS_WORK_LOAD_INDEX


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
        super().__init__(elasticity_handler)

        possible_resolutions = elasticity_handler.state_resolution.possible_states
        possible_fps = elasticity_handler.state_fps.possible_states
        possible_work_loads = elasticity_handler.state_work_load.possible_states

        # Define action spaces for each state dimension
        self.resolution_actions = list(ActionType)
        self.fps_actions = list(ActionType)
        self.work_load_actions = list(ActionType)

        # Control dimensions for each state factor
        self.control_dims = [
            len(self.resolution_actions),
            len(self.fps_actions),
            len(self.work_load_actions)
        ]

        # Define state dimensions
        self.num_resolution_states = len(possible_resolutions)
        self.num_fps_states = len(possible_fps)
        self.num_work_load_states = len(possible_work_loads)

        # SLO states
        self.num_queue_states = len(SloStatus)
        self.num_memory_states = len(SloStatus)
        self.num_slo = 2

        # Policy settings
        self.policy_length = policy_length

        self._setup_generative_model()

    def step(self) -> tuple[list[ActionType], bool]:
        """
        Perform a single step of the active inference loop
        Returns tuple of (actions_taken, success_status)
        """
        observations = self._get_observations()

        # Run active inference
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()

        # Sample actions for each control dimension
        action_indices = np.array(self.agent.sample_action(), dtype=int).tolist()
        print(f'sampled actions: {action_indices}')
        actions = [
            self.resolution_actions[action_indices[ActiveInferenceAgentExperimental.ACTION_RESOLUTION_INDEX]],
            self.fps_actions[action_indices[ActiveInferenceAgentExperimental.ACTION_FPS_INDEX]],
            self.work_load_actions[action_indices[ActiveInferenceAgentExperimental.ACTION_WORK_LOAD_INDEX]]
        ]

        success = self._perform_actions(actions)

        return actions, success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()

    def _setup_generative_model(self):
        """Set up the generative model for active inference"""
        # Observation dimensions
        self.obs_dims = [
            self.num_resolution_states,
            self.num_fps_states,
            self.num_work_load_states,
            self.num_queue_states,
            self.num_memory_states,
        ]

        # State dimensions
        self.state_dims = [
            self.num_resolution_states,
            self.num_fps_states,
            self.num_work_load_states,
        ]

        # Build model components
        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        self.agent = Agent(A=A, B=B, C=C, D=D,
                           num_controls=self.control_dims,
                           policy_len=self.policy_length,
                           control_fac_idx=[0, 1, 2])

    def _construct_A_matrix(self):
        """Construct observation model (likelihood)"""
        A = utils.obj_array(len(self.obs_dims))

        # Initialize each observation modality array
        for obs_idx in range(len(self.obs_dims)):
            # Get observation dimension and state dimensions
            obs_dim = self.obs_dims[obs_idx]
            state_dims = self.state_dims
            # Create array with shape (obs_dim, *state_dims)
            A[obs_idx] = np.zeros((obs_dim,) + tuple(state_dims))

        # Resolution observation
        for i in range(self.num_resolution_states):
            A[self.OBS_RESOLUTION_INDEX][i, i, :, :] = 1.0

        # FPS observation
        for i in range(self.num_fps_states):
            A[self.OBS_FPS_INDEX][i, :, i, :] = 1.0

        # Workload observation
        for i in range(self.num_work_load_states):
            A[self.OBS_WORK_LOAD_INDEX][i, :, :, i] = 1.0

        # Queue SLO observation
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    queue_probs = self.slo_manager.get_qsize_slo_state_probabilities()
                    A[self.OBS_QUEUE_SIZE_INDEX][:, res_idx, fps_idx, wl_idx] = queue_probs

        # Memory SLO observation
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for wl_idx in range(self.num_work_load_states):
                    memory_probs = self.slo_manager.get_mem_slo_state_probabilities()
                    A[self.OBS_MEMORY_USAGE_INDEX][:, res_idx, fps_idx, wl_idx] = memory_probs

        return A

    def _construct_B_matrix(self):
        """Construct transition model"""
        B = utils.obj_array(len(self.state_dims))
        transition_prob = 0.9  # Probability of successful state transition

        # Resolution transitions
        B[ActiveInferenceAgentExperimental.OBS_RESOLUTION_INDEX] = self._create_factor_transition_matrix(
            self.num_resolution_states,
            len(self.resolution_actions),
            ActionType.INCREASE.value,
            ActionType.DECREASE.value,
            transition_prob
        )

        # FPS transitions
        B[ActiveInferenceAgentExperimental.OBS_FPS_INDEX] = self._create_factor_transition_matrix(
            self.num_fps_states,
            len(self.fps_actions),
            ActionType.INCREASE.value,
            ActionType.DECREASE.value,
            transition_prob
        )

        # Workload transitions
        B[ActiveInferenceAgentExperimental.OBS_WORK_LOAD_INDEX] = self._create_factor_transition_matrix(
            self.num_work_load_states,
            len(self.work_load_actions),
            ActionType.INCREASE.value,
            ActionType.DECREASE.value,
            transition_prob
        )

        return B

    def _create_factor_transition_matrix(self, num_states, num_actions, increase_idx, decrease_idx, transition_prob):
        """Create transition matrix for a single state factor"""
        B = np.zeros((num_states, num_states, num_actions))

        for action in range(num_actions):
            for state in range(num_states):
                if action == increase_idx:
                    if state < num_states - 1:
                        B[state + 1, state, action] = transition_prob
                        B[state, state, action] = 1 - transition_prob
                    else:
                        B[state, state, action] = 1.0
                elif action == decrease_idx:
                    if state > 0:
                        B[state - 1, state, action] = transition_prob
                        B[state, state, action] = 1 - transition_prob
                    else:
                        B[state, state, action] = 1.0
                else:
                    B[state, state, action] = 1.0
        return B

    def _construct_C_matrix(self):
        """Construct preference model"""
        C = utils.obj_array(len(self.obs_dims))

        # Initialize each observation modality array
        for obs_idx in range(len(self.obs_dims)):
            obs_dim = self.obs_dims[obs_idx]
            C[obs_idx] = np.zeros(obs_dim)

        # Resolution preferences
        for i in range(self.num_resolution_states):
            if self.num_resolution_states > 1:
                C[self.OBS_RESOLUTION_INDEX][i] = i * self.LOW_PREFERENCE / (self.num_resolution_states - 1)
            else:
                C[self.OBS_RESOLUTION_INDEX][i] = 0.0

        # FPS preferences
        for i in range(self.num_fps_states):
            if self.num_fps_states > 1:
                C[self.OBS_FPS_INDEX][i] = i * self.LOW_PREFERENCE / (self.num_fps_states - 1)
            else:
                C[self.OBS_FPS_INDEX][i] = 0.0

        # Workload preferences
        for i in range(self.num_work_load_states):
            if self.num_work_load_states > 1:
                C[self.OBS_WORK_LOAD_INDEX][i] = i * self.VERY_LOW_PREFERENCE / (self.num_work_load_states - 1)
            else:
                C[self.OBS_WORK_LOAD_INDEX][i] = 0.0

        # Queue SLO preferences
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.OK.value] = self.VERY_STRONG_PREFERENCE
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value] = self.VERY_STRONG_AVERSION

        # Memory SLO preferences
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.OK.value] = self.VERY_STRONG_PREFERENCE
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value] = self.NEUTRAL
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value] = self.VERY_STRONG_AVERSION

        return C

    def _construct_D_matrix(self):
        """Construct initial state prior"""
        D = utils.obj_array(len(self.state_dims))

        # Initialize with current states
        for i, state in enumerate([
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index
        ]):
            D[i] = np.zeros(self.state_dims[i])
            D[i][state] = 1.0

        return D

    def _get_observations(self) -> list[int]:
        """Get current system observations"""
        queue_status, memory_status = self.slo_manager.get_all_slo_status(track_stats=True)
        return [
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index,
            queue_status.value,
            memory_status.value
        ]

    def _perform_actions(self, actions: list[ActionType]) -> bool:
        """Execute actions with SLO-aware validation"""
        success = True

        # Resolution action
        if actions[ActiveInferenceAgentExperimental.ACTION_RESOLUTION_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_resolution()
        elif actions[ActiveInferenceAgentExperimental.ACTION_RESOLUTION_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_resolution()

        # FPS action
        if actions[ActiveInferenceAgentExperimental.ACTION_FPS_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_fps()
        elif actions[ActiveInferenceAgentExperimental.ACTION_FPS_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_fps()

        # Workload action
        if actions[ActiveInferenceAgentExperimental.ACTION_WORK_LOAD_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_work_load()
        elif actions[ActiveInferenceAgentExperimental.ACTION_WORK_LOAD_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_work_load()

        return success