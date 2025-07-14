import logging

import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.action.action_type import ActionType
from producer.elasticity.action.general_action_type import GeneralActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.view.aif_agent_observations import AIFAgentObservations
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ActiveInferenceAgentRelativeControl(ElasticityAgent):
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

        self.observations = AIFAgentObservations(elasticity_handler.observations(), self.slo_manager)

        # Get the relative actions view for clean interface to increase/decrease actions
        self.actions = elasticity_handler.actions_relative()

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
        
        # Learning settings
        self.learning_rate_A = 1.0
        self.learning_rate_B = 1.0

        self._setup_generative_model()

    def step(self):
        """
        Perform a single step of the active inference loop with learning.

        Returns:
            tuple: (ActionType, bool) Action type and whether the actions were successful
        """
        # Get current observations
        observations = self.observations.get_observations()
        
        # Store previous beliefs for learning (if available)
        prev_beliefs = None
        if hasattr(self.agent, 'qs'):
            prev_beliefs = [qs.copy() for qs in self.agent.qs]

        # Perform active inference
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()

        actions_idx = np.array(self.agent.sample_action(), dtype=int).tolist()
        log.debug(f'AIF Agent sampled actions: resolution_action={actions_idx[0]}, fps_action={actions_idx[1]}, inference_quality_action={actions_idx[2]}')

        # Execute actions
        success = self._perform_actions(actions_idx)
        
        # Update A matrix with learning (always enabled)
        if prev_beliefs is not None:
            # Update A matrix based on observed outcomes
            self.agent.update_A(observations)
        
        return ActionType.NONE, success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()

    def _setup_generative_model(self):
        """
        Set up the generative model for active inference
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
            A_factor_list=[
                [0],        # OBS_RESOLUTION_INDEX depends only on state factor 0 (resolution)
                [1],        # OBS_FPS_INDEX depends only on state factor 1 (FPS)
                [2],        # OBS_INFERENCE_QUALITY_INDEX depends only on state factor 2 (inference quality)
                [0, 1, 2],  # OBS_QUEUE_SIZE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_MEMORY_USAGE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_GLOBAL_PROCESSING_TIME_INDEX depends on all state factors
                [0, 1, 2]   # OBS_WORKER_PROCESSING_TIME_INDEX depends on all state factors
            ],
            B_factor_list=[[0], [1], [2]],  # Each B matrix controls its corresponding state factor
            pA=utils.dirichlet_like(A),     # Initialize Dirichlet priors using built-in function
            lr_pA=self.learning_rate_A,     # Learning rate for A matrix
            control_fac_idx=[0, 1, 2],      # indices of hidden state factors that are directly controllable
        )

    def _construct_A_matrix(self):
        """
        Construct the A matrix (observation model) - Likelihood mapping from hidden states to observations

        The A array encodes the likelihood mapping between hidden states and observations.
        The A array answers: "Given a hidden state, what observation am I likely to see?"
        
        When using A_factor_list, each A matrix only has dimensions for the
        state factors it depends on, not all state factors.
        """
        A = utils.obj_array(len(self.obs_dims))

        # A[0]: Resolution observation - depends only on resolution state (factor 0)
        A[self.OBS_RESOLUTION_INDEX] = np.zeros((self.num_resolution_states, self.num_resolution_states))
        for i in range(self.num_resolution_states):
            A[self.OBS_RESOLUTION_INDEX][i, i] = 1.0

        # A[1]: FPS observation - depends only on FPS state (factor 1)
        A[self.OBS_FPS_INDEX] = np.zeros((self.num_fps_states, self.num_fps_states))
        for i in range(self.num_fps_states):
            A[self.OBS_FPS_INDEX][i, i] = 1.0

        # A[2]: Inference Quality observation - depends only on inference quality state (factor 2)
        A[self.OBS_INFERENCE_QUALITY_INDEX] = np.zeros((self.num_inference_quality_states, self.num_inference_quality_states))
        for i in range(self.num_inference_quality_states):
            A[self.OBS_INFERENCE_QUALITY_INDEX][i, i] = 1.0

        # SLO observations: depend on all three state factors [0, 1, 2]
        slo_shape = (self.num_queue_states, self.num_resolution_states, self.num_fps_states, self.num_inference_quality_states)
        
        # Initialize SLO observation matrices
        A[self.OBS_QUEUE_SIZE_INDEX] = np.zeros(slo_shape)
        A[self.OBS_MEMORY_USAGE_INDEX] = np.zeros(slo_shape)
        A[self.OBS_GLOBAL_PROCESSING_TIME_INDEX] = np.zeros(slo_shape)
        A[self.OBS_WORKER_PROCESSING_TIME_INDEX] = np.zeros(slo_shape)

        # SLO probability mappings: All 4 SLO status observations
        # Instead of using constant probabilities, we model how different parameter combinations
        # affect SLO violations. Higher quality parameters (resolution, fps, inference quality)
        # should increase the probability of SLO violations (CRITICAL/WARNING states)
        
        for res in range(self.num_resolution_states):
            for fps in range(self.num_fps_states):
                for wl in range(self.num_inference_quality_states):
                    # Calculate normalized load factor (0.0 to 1.0)
                    # Higher indices = higher quality = higher computational load = higher SLO violation probability
                    res_load = res / max(1, self.num_resolution_states - 1)
                    fps_load = fps / max(1, self.num_fps_states - 1)
                    wl_load = wl / max(1, self.num_inference_quality_states - 1)
                    
                    # Weighted combination (inference quality has highest impact)
                    # Reduced the impact to be less aggressive about predicting SLO violations
                    combined_load = (res_load * 0.2 + fps_load * 0.2 + wl_load * 0.3)
                    
                    # Convert to SLO violation probabilities with more conservative mapping
                    # Only predict violations at higher load levels
                    if combined_load < 0.3:
                        # Low load: very high chance of OK SLOs
                        p_ok = 0.95
                        p_critical = 0.01
                        p_warning = 0.04
                    elif combined_load < 0.6:
                        # Medium load: good chance of OK, some warnings
                        p_ok = 0.8
                        p_critical = 0.05
                        p_warning = 0.15
                    elif combined_load < 0.8:
                        # High load: more warnings, some critical
                        p_ok = 0.6
                        p_critical = 0.2
                        p_warning = 0.2
                    else:
                        # Very high load: likely violations
                        p_ok = 0.3
                        p_critical = 0.5
                        p_warning = 0.2
                    
                    # Normalize probabilities (should already sum to 1, but just in case)
                    total = p_ok + p_warning + p_critical
                    slo_probs = [p_ok/total, p_warning/total, p_critical/total]
                    
                    # Apply to all SLO observation types
                    A[self.OBS_QUEUE_SIZE_INDEX][:, res, fps, wl] = slo_probs
                    A[self.OBS_MEMORY_USAGE_INDEX][:, res, fps, wl] = slo_probs
                    A[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][:, res, fps, wl] = slo_probs
                    A[self.OBS_WORKER_PROCESSING_TIME_INDEX][:, res, fps, wl] = slo_probs

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
        B[0] = self._construct_sub_transition_model(self.num_resolution_states,
                                                     self.num_resolution_actions,
                                                     probability_to_change_state)

        # State 1 - FPS: Transitions for FPS states based on actions
        B[1] = self._construct_sub_transition_model(self.num_fps_states,
                                                     self.num_fps_actions,
                                                     probability_to_change_state)

        # State 2 - Work Load: Transitions for Work load states based on actions
        B[2] = self._construct_sub_transition_model(self.num_inference_quality_states,
                                                     self.num_work_load_actions,
                                                     probability_to_change_state)

        return B

    @staticmethod
    def _construct_sub_transition_model(num_states: int, num_actions: int, probability_to_change_state: float):
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

        self._set_quality_preferences(C)
        self._set_slo_preferences(C)

        return C

    def _set_quality_preferences(self, C):
        """Set preferences for quality parameters (resolution, FPS, inference quality)"""
        # Resolution preferences - strong preference for higher quality
        if self.num_resolution_states > 1:
            C[self.OBS_RESOLUTION_INDEX][:] = [
                self.STRONG_PREFERENCE * (i / (self.num_resolution_states - 1))
                for i in range(self.num_resolution_states)
            ]

        # FPS preferences - strong preference for higher quality  
        if self.num_fps_states > 1:
            C[self.OBS_FPS_INDEX][:] = [
                self.MEDIUM_PREFERENCE * (i / (self.num_fps_states - 1))
                for i in range(self.num_fps_states)
            ]

        # Inference Quality preferences - medium preference (still expensive but important)
        if self.num_inference_quality_states > 1:
            C[self.OBS_INFERENCE_QUALITY_INDEX][:] = [
                self.MEDIUM_PREFERENCE * (i / (self.num_inference_quality_states - 1))
                for i in range(self.num_inference_quality_states)
            ]

    def _set_slo_preferences(self, C):
        """Set identical preferences for all SLO observations"""
        slo_indices = [
            self.OBS_QUEUE_SIZE_INDEX,
            self.OBS_MEMORY_USAGE_INDEX,
            self.OBS_GLOBAL_PROCESSING_TIME_INDEX,
            self.OBS_WORKER_PROCESSING_TIME_INDEX
        ]

        for slo_idx in slo_indices:
            C[slo_idx][SloStatus.OK.value] = self.MEDIUM_PREFERENCE
            C[slo_idx][SloStatus.WARNING.value] = self.NEUTRAL  # Slight aversion to warnings
            C[slo_idx][SloStatus.CRITICAL.value] = self.STRONG_AVERSION

    def _construct_D_matrix(self):
        """Construct the D matrix (prior believes over states) - Initial beliefs, i.e. what states are expected before making an observation"""
        D = utils.obj_array(len(self.state_dims))

        current_states = self.observations.get_states_indices()

        for i, state in enumerate(current_states):
            D[i] = np.zeros(self.state_dims[i])
            D[i][state] = 1.0

        return D

    def _perform_actions(self, actions: list[int]):
        """Tries to perform for all state dimensions using the relative actions view"""
        # TODO: Return list of actions and if they were successful
        success = True
        # Resolution action
        if actions[self.ACTION_RESOLUTION_INDEX] == ActionType.INCREASE:
            success &= self.actions.increase_resolution()
        elif actions[self.ACTION_RESOLUTION_INDEX] == ActionType.DECREASE:
            success &= self.actions.decrease_resolution()

        # FPS action
        if actions[self.ACTION_FPS_INDEX] == ActionType.INCREASE:
            success &= self.actions.increase_fps()
        elif actions[self.ACTION_FPS_INDEX] == ActionType.DECREASE:
            success &= self.actions.decrease_fps()

        # Workload action
        if actions[self.ACTION_INFERENCE_QUALITY_INDEX] == ActionType.INCREASE:
            success &= self.actions.increase_inference_quality()
        elif actions[self.ACTION_INFERENCE_QUALITY_INDEX] == ActionType.DECREASE:
            success &= self.actions.decrease_inference_quality()

        return success