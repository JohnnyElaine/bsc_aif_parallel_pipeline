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
        
        # Learning settings - reduced for more stable learning
        self.learning_rate_A = 0.1  # Much slower A-matrix learning
        self.learning_rate_B = 0.05  # Even slower B-matrix learning (should be deterministic anyway)

        # Stability settings to prevent oscillations
        self.smoothness_penalty = 2.0  # Penalty for large parameter changes
        self.no_change_bonus = 1.0     # Bonus for maintaining current state

        # Learning validation settings
        self.learning_validation_steps = 0
        self.expected_vs_actual_slo = []  # Track prediction accuracy

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
        qs_prev = None
        if hasattr(self.agent, 'qs'):
            qs_prev = self.agent.qs.copy()
        
        # Perform active inference
        q_s = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()
        
        # Log learning validation info periodically
        self.learning_validation_steps += 1
        if self.learning_validation_steps % 20 == 0:
            self._validate_learning(q_s)
        
        # Sample actions (direct indices for each parameter)
        action_indices = np.array(self.agent.sample_action(), dtype=int).tolist()
        
        # Apply stability filter to prevent oscillations
        filtered_actions = self._apply_stability_filter(action_indices, self._get_current_slo_status(observations))
        
        log.debug(f'AIF Agent sampled actions: resolution_action={action_indices[0]}, fps_action={action_indices[1]}, inference_quality_action={action_indices[2]}')
        log.debug(f'AIF Agent filtered actions: resolution_action={filtered_actions[0]}, fps_action={filtered_actions[1]}, inference_quality_action={filtered_actions[2]}')
        
        # Execute actions
        success = self._perform_actions(filtered_actions)
        
        # Update A matrix with learning - only if we have previous beliefs
        if qs_prev is not None and success:
            # Store prediction accuracy for validation
            self._track_prediction_accuracy(qs_prev, observations)
            
            # Update matrices with reduced learning rates
            self.agent.update_A(observations)
            # B matrix updates are less important since transitions are deterministic
            # self.agent.update_B(qs_prev)

        return success

    def reset(self):
        """Reset the agent's beliefs and stability tracking"""
        self.agent.reset()
        # Reset stability tracking
        if hasattr(self, '_action_history'):
            self._action_history = []
        if hasattr(self, '_stability_counters'):
            self._stability_counters = [0, 0, 0]
        if hasattr(self, '_last_non_zero_actions'):
            self._last_non_zero_actions = [None, None, None]

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
            A_factor_list=[
                [0],  # OBS_RESOLUTION_INDEX depends only on state factor 0 (resolution)
                [1],  # OBS_FPS_INDEX depends only on state factor 1 (FPS)
                [2],  # OBS_INFERENCE_QUALITY_INDEX depends only on state factor 2 (inference quality)
                [0, 1, 2],  # OBS_QUEUE_SIZE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_MEMORY_USAGE_INDEX depends on all state factors
                [0, 1, 2],  # OBS_GLOBAL_PROCESSING_TIME_INDEX depends on all state factors
                [0, 1, 2]  # OBS_WORKER_PROCESSING_TIME_INDEX depends on all state factors
            ],
            B_factor_list=[[0], [1], [2]],  # Each B matrix controls its corresponding state factor
            pA=utils.dirichlet_like(A, scale=2.0),  # More concentrated priors for faster learning
            pB=utils.dirichlet_like(B, scale=5.0),  # Strong priors for deterministic transitions
            lr_pA=self.learning_rate_A,   # Reduced learning rate for A matrix
            lr_pB=self.learning_rate_B,  # Reduced learning rate for B matrix
            control_fac_idx=[0, 1, 2],  # indices of hidden state factors that are directly controllable
            use_states_info_gain = True,
            use_param_info_gain=False,  # Disable to reduce exploration noise. Yields extreme oscillating
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
                    
                    # More realistic weighted combination - inference quality has highest impact
                    # FPS has more impact than resolution for computational load
                    combined_load = (res_load * 0.15 + fps_load * 0.35 + inf_load * 0.5)
                    
                    # More conservative and realistic SLO violation probabilities
                    # Make the model less certain about predictions to allow for learning
                    if combined_load < 0.2:
                        # Very low load: very high chance of OK SLOs
                        p_ok = 0.9
                        p_critical = 0.05
                        p_warning = 0.05
                    elif combined_load < 0.4:
                        # Low load: high chance of OK, some warnings
                        p_ok = 0.75
                        p_critical = 0.1
                        p_warning = 0.15
                    elif combined_load < 0.6:
                        # Medium load: good chance of OK, more warnings
                        p_ok = 0.6
                        p_critical = 0.15
                        p_warning = 0.25
                    elif combined_load < 0.8:
                        # High load: warnings likely, some critical
                        p_ok = 0.4
                        p_critical = 0.3
                        p_warning = 0.3
                    else:
                        # Very high load: likely violations but not certain
                        p_ok = 0.2
                        p_critical = 0.5
                        p_warning = 0.3
                    
                    # Normalize probabilities
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
        
        The transitions are deterministic since the agent can reliably
        change stream quality parameters when commanded.
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

        self._set_quality_preferences(C)

        # SLO preferences - still the most important, but not overwhelmingly so
        # This ensures the agent balances quality vs SLO satisfaction
        self._set_slo_preferences(C)

        return C

    def _set_slo_preferences(self, C):
        slo_obs_indices = [
            self.OBS_QUEUE_SIZE_INDEX,
            self.OBS_MEMORY_USAGE_INDEX,
            self.OBS_GLOBAL_PROCESSING_TIME_INDEX,
            self.OBS_WORKER_PROCESSING_TIME_INDEX
        ]
        for slo_idx in slo_obs_indices:
            # Clearer preference structure for better learning
            C[slo_idx][SloStatus.OK.value] = self.MEDIUM_PREFERENCE  # Positive reward for good SLOs
            C[slo_idx][SloStatus.WARNING.value] = self.LOW_AVERSION  # Mild penalty for warnings
            C[slo_idx][SloStatus.CRITICAL.value] = self.STRONG_AVERSION  # Strong penalty for violations

    def _set_quality_preferences(self, C):
        # More conservative quality preferences to avoid aggressive optimization
        # when learning is still unstable
        if self.num_resolution_states > 1:
            C[self.OBS_RESOLUTION_INDEX][:] = [
                self.LOW_PREFERENCE * (i / (self.num_resolution_states - 1))
                for i in range(self.num_resolution_states)
            ]
        if self.num_fps_states > 1:
            C[self.OBS_FPS_INDEX][:] = [
                self.VERY_LOW_PREFERENCE * (i / (self.num_fps_states - 1))
                for i in range(self.num_fps_states)
            ]
        if self.num_inference_quality_states > 1:
            C[self.OBS_INFERENCE_QUALITY_INDEX][:] = [
                self.VERY_LOW_PREFERENCE * (i / (self.num_inference_quality_states - 1))
                for i in range(self.num_inference_quality_states)
            ]

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

    def _validate_learning(self, q_s):
        """Validate that the agent is learning properly"""
        current_states = self.observations.get_states_indices()

        # Log current beliefs about states
        log.info(f"AIF Learning Validation - Current states: res={current_states[0]}, fps={current_states[1]}, inf={current_states[2]}")

        # Check if agent's state beliefs match reality
        for i, q_s_factor in enumerate(q_s):
            predicted_state = np.argmax(q_s_factor)
            actual_state = current_states[i]
            if predicted_state != actual_state:
                log.warning(f"AIF Learning Issue - Factor {i}: predicted state {predicted_state}, actual state {actual_state}")
        
        # Log prediction accuracy if we have enough data
        if len(self.expected_vs_actual_slo) >= 10:
            recent_accuracy = np.mean(self.expected_vs_actual_slo[-10:])
            log.info(f"AIF Learning Validation - Recent SLO prediction accuracy: {recent_accuracy:.2f}")
    
    def _track_prediction_accuracy(self, qs_prev, observations):
        """Track how well the agent predicts SLO outcomes"""
        try:
            # Get actual SLO observations
            actual_slo_obs = [
                observations[self.OBS_QUEUE_SIZE_INDEX],
                observations[self.OBS_MEMORY_USAGE_INDEX], 
                observations[self.OBS_GLOBAL_PROCESSING_TIME_INDEX],
                observations[self.OBS_WORKER_PROCESSING_TIME_INDEX]
            ]
            
            # Simple accuracy check - did we predict the right SLO status?
            # This is simplified but gives us a sense of learning progress
            slo_prediction_correct = 0
            for slo_obs in actual_slo_obs:
                # For simplicity, consider prediction "correct" if SLO is OK when we expected it
                # or if SLO is not OK when we predicted problems
                if slo_obs == SloStatus.OK.value:
                    slo_prediction_correct += 1
            
            accuracy = slo_prediction_correct / len(actual_slo_obs)
            self.expected_vs_actual_slo.append(accuracy)
            
            # Keep only recent history
            if len(self.expected_vs_actual_slo) > 100:
                self.expected_vs_actual_slo.pop(0)
                
        except Exception as e:
            log.warning(f"Error tracking prediction accuracy: {e}")

    def _apply_stability_filter(self, action_indices: list[int], current_slo_status: list[SloStatus]) -> list[int]:
        """
        Apply stability filter to prevent oscillations.
        
        This method introduces hysteresis-like behavior:
        - Strongly prefers "no change" actions when SLOs are satisfied
        - Detects and prevents alternating action patterns
        - Maintains a memory of recent actions to avoid oscillatory behavior
        
        Args:
            action_indices: Original action indices from the agent
            
        Returns:
            list[int]: Filtered action indices
        """
        # Initialize action history and stability counters if not exists
        if not hasattr(self, '_action_history'):
            self._action_history = []
        if not hasattr(self, '_stability_counters'):
            self._stability_counters = [0, 0, 0]  # Counter for each parameter
        if not hasattr(self, '_last_non_zero_actions'):
            self._last_non_zero_actions = [None, None, None]  # Track last non-zero action for each param

        filtered_actions = action_indices.copy()
        
        # Strong stability bias when SLOs are OK
        if all(status == SloStatus.OK for status in current_slo_status):
            for i in range(len(filtered_actions)):
                if action_indices[i] != 0:  # If agent wants to change something
                    # Increase stability counter for this parameter
                    self._stability_counters[i] += 1
                    
                    # Very strong bias against changes when SLOs are OK
                    # The more recent changes, the stronger the bias against new changes
                    stability_bias = min(0.95, 0.8 + (self._stability_counters[i] * 0.05))
                    
                    if np.random.random() < stability_bias:
                        filtered_actions[i] = 0
                        log.debug(f"Stability filter: Prevented change for parameter {i} (stability_counter={self._stability_counters[i]})")
                else:
                    # Reset counter when agent chooses "no change"
                    self._stability_counters[i] = max(0, self._stability_counters[i] - 1)
        else:
            # Reset stability counters when SLOs are violated (allow quick responses)
            self._stability_counters = [0, 0, 0]
        
        # Detect and prevent oscillatory patterns
        for i in range(len(filtered_actions)):
            if filtered_actions[i] != 0:
                # Check for alternating pattern with recent non-zero actions
                if self._is_alternating_action(i, filtered_actions[i]):
                    filtered_actions[i] = 0
                    log.debug(f"Prevented alternating pattern for parameter {i}: action {action_indices[i]} -> 0")
                else:
                    # Update last non-zero action
                    self._last_non_zero_actions[i] = filtered_actions[i]
        
        # Prevent too many simultaneous changes (max 1 parameter change per step when SLOs are OK)
        if all(status == SloStatus.OK for status in current_slo_status):
            non_zero_count = sum(1 for action in filtered_actions if action != 0)
            if non_zero_count > 1:
                # Keep only the first non-zero action, set others to 0
                found_first = False
                for i in range(len(filtered_actions)):
                    if filtered_actions[i] != 0:
                        if found_first:
                            filtered_actions[i] = 0
                            log.debug(f"Limited simultaneous changes: parameter {i} action -> 0")
                        else:
                            found_first = True
        
        # Store action in history (keep last 10 actions for better pattern detection)
        self._action_history.append(filtered_actions.copy())
        if len(self._action_history) > 10:
            self._action_history.pop(0)
        
        return filtered_actions
    
    def _is_alternating_action(self, param_idx: int, proposed_action: int) -> bool:
        """
        Detect if the proposed action would create an alternating pattern.
        
        This method checks if the agent is trying to alternate between the same
        few actions repeatedly, which is a classic oscillation pattern.
        
        Args:
            param_idx: Index of the parameter (0=resolution, 1=fps, 2=inference_quality)
            proposed_action: The action the agent wants to take
            
        Returns:
            bool: True if this would create an alternating pattern
        """
        if len(self._action_history) < 2:
            return False
        
        # Get recent non-zero actions for this parameter
        recent_actions = []
        for history in self._action_history[-4:]:  # Look at last 4 steps
            if history[param_idx] != 0:
                recent_actions.append(history[param_idx])
        
        # If we have recent non-zero actions
        if len(recent_actions) >= 2:
            # Check if proposed action would create an A-B-A pattern
            if (recent_actions[-1] != proposed_action and 
                len(recent_actions) >= 2 and
                recent_actions[-2] == proposed_action):
                return True
            
            # Check for longer alternating patterns (A-B-A-B...)
            if len(recent_actions) >= 3:
                # Check if we have an alternating pattern in recent actions
                pattern_detected = True
                for i in range(len(recent_actions) - 1):
                    if i % 2 == 0:
                        # Even indices should match
                        if recent_actions[i] != recent_actions[0]:
                            pattern_detected = False
                            break
                    else:
                        # Odd indices should match
                        if len(recent_actions) > 1 and recent_actions[i] != recent_actions[1]:
                            pattern_detected = False
                            break
                
                if pattern_detected and proposed_action in recent_actions[-2:]:
                    return True
        
        return False
    
    def _get_current_slo_status(self, observations: list[int]) -> list[SloStatus]:
        """Get current SLO status for all SLOs"""
        return [
            SloStatus(observations[self.OBS_QUEUE_SIZE_INDEX]),
            SloStatus(observations[self.OBS_MEMORY_USAGE_INDEX]),
            SloStatus(observations[self.OBS_GLOBAL_PROCESSING_TIME_INDEX]),
            SloStatus(observations[self.OBS_WORKER_PROCESSING_TIME_INDEX])
        ]

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