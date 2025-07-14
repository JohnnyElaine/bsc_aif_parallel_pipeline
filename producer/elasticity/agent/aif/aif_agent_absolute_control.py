import logging

import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.view.aif_agent_observations import AIFAgentObservations
from producer.request_handling.request_handler import RequestHandler
from producer.task_generation.task_generator import TaskGenerator

log = logging.getLogger('producer')


class ActiveInferenceAgentAbsoluteControl(ElasticityAgent):
    """
    Active Inference Agent using absolute control for distributed video processing system.
    
    This agent directly sets stream parameter indices (resolution, FPS, inference quality)
    to optimize Quality of Experience while maintaining Service Level Objectives.
    
    Key differences from relative control:
    - Directly sets parameter indices instead of incremental changes
    - Actions represent absolute target states rather than relative movements
    - More efficient for dramatic quality adjustments when needed
    """
    
    # Observation indices
    OBS_RESOLUTION_INDEX = 0
    OBS_FPS_INDEX = 1
    OBS_INFERENCE_QUALITY_INDEX = 2
    OBS_QUEUE_SIZE_INDEX = 3
    OBS_MEMORY_USAGE_INDEX = 4
    OBS_GLOBAL_PROCESSING_TIME_INDEX = 5
    OBS_WORKER_PROCESSING_TIME_INDEX = 6

    # Preference values
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

    def __init__(self,
                 elasticity_handler: ElasticityHandler,
                 request_handler: RequestHandler,
                 task_generator: TaskGenerator,
                 policy_length: int = 1,
                 track_slo_stats=True):
        super().__init__(elasticity_handler, request_handler, task_generator, track_slo_stats=track_slo_stats)

        self.observations = AIFAgentObservations(elasticity_handler.observations(), self.slo_manager)
        self.actions = elasticity_handler.actions_absolute()

        self.num_resolution_states = len(elasticity_handler.state_resolution.possible_states)
        self.num_fps_states = len(elasticity_handler.state_fps.possible_states)
        self.num_inference_quality_states = len(elasticity_handler.state_inference_quality.possible_states)

        # For absolute control with "no change" action:
        # Action 0 = no change, Actions 1-N = set to parameter index 0-(N-1)
        self.num_resolution_actions = self.num_resolution_states + 1  # +1 for "no change"
        self.num_fps_actions = self.num_fps_states + 1  # +1 for "no change"
        self.num_inference_quality_actions = self.num_inference_quality_states + 1  # +1 for "no change"

        # Action space dimensions for pymdp
        self.control_dims = [
            self.num_resolution_actions,
            self.num_fps_actions,
            self.num_inference_quality_actions
        ]

        # SLO observation dimensions
        self.num_slo_states = len(SloStatus)  # OK, WARNING, CRITICAL

        # Policy settings
        self.policy_length = policy_length
        
        # Learning settings
        self.learning_rate_A = 1.0
        self.learning_rate_B = 1.0

        self._setup_generative_model()

    def step(self):
        """
        Perform a single step of the active inference loop with learning.
        
        Returns:
            bool: Whether the actions were successful
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
        
        # Sample actions (direct indices for each parameter)
        action_indices = np.array(self.agent.sample_action(), dtype=int).tolist()
        log.debug(f'AIF Agent sampled actions: resolution_action={action_indices[0]}, fps_action={action_indices[1]}, inference_quality_action={action_indices[2]}')
        
        # Execute actions
        success = self._perform_actions(action_indices)
        
        # Update A matrix with learning (always enabled)
        if prev_beliefs is not None:
            # Update A matrix based on observed outcomes
            self.agent.update_A(observations)

        return success

    def reset(self):
        """Reset the agent's beliefs"""
        self.agent.reset()

    def _setup_generative_model(self):
        """Set up the generative model for active inference"""
        
        # Observation dimensions
        self.obs_dims = [
            self.num_resolution_states,            # Resolution state
            self.num_fps_states,                  # FPS state
            self.num_inference_quality_states,    # Inference quality state
            self.num_slo_states,                  # Queue SLO status
            self.num_slo_states,                  # Memory SLO status
            self.num_slo_states,                  # Global processing time SLO status
            self.num_slo_states                   # Worker processing time SLO status
        ]

        # Hidden state dimensions (what the agent can control)
        self.state_dims = [
            self.num_resolution_states,
            self.num_fps_states,
            self.num_inference_quality_states
        ]

        # Construct matrices
        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        # Create pymdp agent with proper factor specifications
        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            num_controls=self.control_dims,
            policy_len=self.policy_length,
            B_factor_list=[[0], [1], [2]],  # Each B matrix controls its corresponding state factor
            A_factor_list=[
                [0],        # OBS_RESOLUTION_INDEX depends only on state factor 0 (resolution)
                [1],        # OBS_FPS_INDEX depends only on state factor 1 (FPS)
                [2],        # OBS_INFERENCE_QUALITY_INDEX depends only on state factor 2 (inference quality)
                [0, 1, 2],  # OBS_QUEUE_SIZE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_MEMORY_USAGE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_GLOBAL_PROCESSING_TIME_INDEX depends on all state factors
                [0, 1, 2]   # OBS_WORKER_PROCESSING_TIME_INDEX depends on all state factors
            ],
            # Learning configuration
            use_param_info_gain=False,  # Disable parameter info gain for simplicity
            lr_pA=self.learning_rate_A,   # Learning rate for A matrix
            pA=utils.dirichlet_like(A),  # Initialize Dirichlet priors using built-in function
            modalities_to_learn='all'   # Learn all observation modalities
        )

    def _construct_A_matrix(self):
        """
        Construct the observation model (A matrix).
        
        Maps from hidden states (resolution, fps, inference_quality) to observations.
        The key insight: higher quality parameters increase computational load,
        which increases the probability of SLO violations.
        
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
        slo_shape = (self.num_slo_states, self.num_resolution_states, self.num_fps_states, self.num_inference_quality_states)
        
        # Initialize SLO observation matrices
        A[self.OBS_QUEUE_SIZE_INDEX] = np.zeros(slo_shape)
        A[self.OBS_MEMORY_USAGE_INDEX] = np.zeros(slo_shape)
        A[self.OBS_GLOBAL_PROCESSING_TIME_INDEX] = np.zeros(slo_shape)
        A[self.OBS_WORKER_PROCESSING_TIME_INDEX] = np.zeros(slo_shape)

        # Fill SLO observations based on computational load
        for res_idx in range(self.num_resolution_states):
            for fps_idx in range(self.num_fps_states):
                for inf_idx in range(self.num_inference_quality_states):
                    
                    # Calculate computational load based on parameter indices
                    # Higher indices = higher quality = higher computational load
                    res_load = res_idx / max(1, self.num_resolution_states - 1)
                    fps_load = fps_idx / max(1, self.num_fps_states - 1)
                    inf_load = inf_idx / max(1, self.num_inference_quality_states - 1)
                    
                    # Weighted combination (inference quality has highest impact)
                    # Reduced the impact to be less aggressive about predicting SLO violations
                    combined_load = (res_load * 0.2 + fps_load * 0.2 + inf_load * 0.3)
                    
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
                    A[self.OBS_QUEUE_SIZE_INDEX][:, res_idx, fps_idx, inf_idx] = slo_probs
                    A[self.OBS_MEMORY_USAGE_INDEX][:, res_idx, fps_idx, inf_idx] = slo_probs
                    A[self.OBS_GLOBAL_PROCESSING_TIME_INDEX][:, res_idx, fps_idx, inf_idx] = slo_probs
                    A[self.OBS_WORKER_PROCESSING_TIME_INDEX][:, res_idx, fps_idx, inf_idx] = slo_probs

        return A

    def _construct_B_matrix(self):
        """
        Construct the transition model (B matrix).
        
        For absolute control with "no change" action:
        - Action 0: No change (stay in current state)
        - Action 1: Set to parameter index 0
        - Action 2: Set to parameter index 1
        - ...
        - Action N: Set to parameter index N-1
        """
        B = utils.obj_array(len(self.state_dims))

        # Resolution transitions
        B[0] = np.zeros((self.num_resolution_states, self.num_resolution_states, self.num_resolution_actions))
        for current_state in range(self.num_resolution_states):
            # Action 0: No change (stay in current state)
            B[0][current_state, current_state, 0] = 1.0
            
            # Actions 1-N: Set to specific parameter index (action-1)
            for action in range(1, self.num_resolution_actions):
                target_state = action - 1  # Convert action to parameter index
                B[0][target_state, current_state, action] = 1.0

        # FPS transitions
        B[1] = np.zeros((self.num_fps_states, self.num_fps_states, self.num_fps_actions))
        for current_state in range(self.num_fps_states):
            # Action 0: No change
            B[1][current_state, current_state, 0] = 1.0
            
            # Actions 1-N: Set to specific parameter index
            for action in range(1, self.num_fps_actions):
                target_state = action - 1
                B[1][target_state, current_state, action] = 1.0

        # Inference Quality transitions
        B[2] = np.zeros((self.num_inference_quality_states, self.num_inference_quality_states, self.num_inference_quality_actions))
        for current_state in range(self.num_inference_quality_states):
            # Action 0: No change
            B[2][current_state, current_state, 0] = 1.0
            
            # Actions 1-N: Set to specific parameter index
            for action in range(1, self.num_inference_quality_actions):
                target_state = action - 1
                B[2][target_state, current_state, action] = 1.0

        return B

    def _construct_C_matrix(self):
        """
        Construct the preference model (C matrix).
        
        Encodes the agent's preferences over observations:
        - Quality parameters: Higher is better (important for user experience)
        - SLO status: Strong preference for OK, strong aversion to CRITICAL
        
        The key insight: When SLOs are satisfied, we want to maximize quality.
        When SLOs are violated, avoiding violations becomes more important than quality.
        """
        C = utils.obj_array(len(self.obs_dims))

        # Initialize preferences
        for obs_idx, obs_dim in enumerate(self.obs_dims):
            C[obs_idx] = np.zeros(obs_dim)

        # Quality parameter preferences - significantly increased to encourage higher quality
        # when SLOs allow it
        
        # Resolution preferences - strong preference for higher quality
        if self.num_resolution_states > 1:
            C[self.OBS_RESOLUTION_INDEX][:] = [
                self.STRONG_PREFERENCE * (i / (self.num_resolution_states - 1))
                for i in range(self.num_resolution_states)
            ]

        # FPS preferences - strong preference for higher quality  
        if self.num_fps_states > 1:
            C[self.OBS_FPS_INDEX][:] = [
                self.STRONG_PREFERENCE * (i / (self.num_fps_states - 1))
                for i in range(self.num_fps_states)
            ]

        # Inference Quality preferences - medium preference (still expensive but important)
        if self.num_inference_quality_states > 1:
            C[self.OBS_INFERENCE_QUALITY_INDEX][:] = [
                self.MEDIUM_PREFERENCE * (i / (self.num_inference_quality_states - 1))
                for i in range(self.num_inference_quality_states)
            ]

        # SLO preferences - still the most important, but not overwhelmingly so
        # This ensures the agent balances quality vs SLO satisfaction
        slo_obs_indices = [
            self.OBS_QUEUE_SIZE_INDEX,
            self.OBS_MEMORY_USAGE_INDEX,
            self.OBS_GLOBAL_PROCESSING_TIME_INDEX,
            self.OBS_WORKER_PROCESSING_TIME_INDEX
        ]

        for slo_idx in slo_obs_indices:
            C[slo_idx][SloStatus.OK.value] = self.VERY_STRONG_PREFERENCE
            C[slo_idx][SloStatus.WARNING.value] = self.LOW_AVERSION  # Slight aversion to warnings
            C[slo_idx][SloStatus.CRITICAL.value] = self.VERY_STRONG_AVERSION

        return C

    def _construct_D_matrix(self):
        """
        Construct the prior beliefs (D matrix).
        
        Initial beliefs about hidden states based on current system state.
        """
        D = utils.obj_array(len(self.state_dims))

        # Get current state indices
        current_states = self.observations.get_states_indices()

        # Set strong prior beliefs based on current state
        for i, current_state in enumerate(current_states):
            D[i] = np.zeros(self.state_dims[i])
            D[i][current_state] = 1.0

        return D

    def _perform_actions(self, action_indices: list[int]) -> bool:
        """
        Execute actions with "no change" support.
        
        Action mapping:
        - Action 0: No change (keep current parameter value)
        - Action 1: Set parameter to index 0 (lowest quality)
        - Action 2: Set parameter to index 1
        - ...
        - Action N: Set parameter to index N-1 (highest quality)
        
        Args:
            action_indices: [resolution_action, fps_action, inference_quality_action]
            
        Returns:
            bool: True if all actions were successful
        """
        success = True
        
        try:
            # Validate action indices
            if (action_indices[0] >= self.num_resolution_actions or 
                action_indices[1] >= self.num_fps_actions or 
                action_indices[2] >= self.num_inference_quality_actions):
                log.error(f'Invalid action indices: {action_indices}')
                return False
            
            # Resolution action
            if action_indices[0] != 0:
                param_index = action_indices[0] - 1
                result = self.actions.change_resolution_index(param_index)
                success = success and (result is not False)
            
            # FPS action
            if action_indices[1] != 0:
                param_index = action_indices[1] - 1
                result = self.actions.change_fps_index(param_index)
                success = success and (result is not False)
            
            # Inference quality action
            if action_indices[2] != 0:
                param_index = action_indices[2] - 1
                result = self.actions.change_inference_quality_index(param_index)
                success = success and (result is not False)
            
        except Exception as e:
            log.error(f'Error executing actions: {e}')
            success = False

        return success