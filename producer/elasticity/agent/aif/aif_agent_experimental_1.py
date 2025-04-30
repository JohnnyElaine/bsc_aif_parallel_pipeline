import logging
import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent

from producer.elasticity.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.elasticity.slo.slo_status import SloStatus

log = logging.getLogger('producer')


class ActiveInferenceAgentExperimental1(ElasticityAgent):
    # Observation indices
    OBS_RESOLUTION_INDEX = 0
    OBS_FPS_INDEX = 1
    OBS_WORK_LOAD_INDEX = 2
    OBS_QUEUE_SIZE_INDEX = 3
    OBS_MEMORY_USAGE_INDEX = 4

    # Action indices
    ACTION_RESOLUTION_INDEX = 0
    ACTION_FPS_INDEX = 1
    ACTION_WORK_LOAD_INDEX = 2

    # Preference parameters
    STRONG_PREFERENCE = 4.0
    MEDIUM_PREFERENCE = 2.0
    LOW_PREFERENCE = 1.0
    SLO_CRITICAL_AVERSION = -4.0
    SLO_WARNING_AVERSION = -1.0

    def __init__(self, elasticity_handler: ElasticityHandler, policy_length: int = 2):
        super().__init__(elasticity_handler)

        # Initialize state dimensions from handler
        self.num_resolution_states = len(elasticity_handler.state_resolution.possible_states)
        self.num_fps_states = len(elasticity_handler.state_fps.possible_states)
        self.num_work_load_states = len(elasticity_handler.state_work_load.possible_states)

        # Action spaces
        self.resolution_actions = list(ActionType)
        self.fps_actions = list(ActionType)
        self.work_load_actions = list(ActionType)

        self.control_dims = [
            len(self.resolution_actions),
            len(self.fps_actions),
            len(self.work_load_actions)
        ]

        self.policy_length = policy_length
        self._setup_generative_model()

    def step(self) -> tuple[list[ActionType], bool]:
        observations = self._get_observations()
        slo_status = self._get_slo_status(observations)

        # Run active inference
        qs = self.agent.infer_states(observations)
        q_pi, efe = self.agent.infer_policies()

        # Sample actions with SLO-aware adjustment
        action_indices = np.array(self.agent.sample_action(), dtype=int).tolist()
        #actions = self._adjust_actions_based_on_slo(action_indices, slo_status)
        actions = [
            self.resolution_actions[action_indices[self.ACTION_RESOLUTION_INDEX]],
            self.fps_actions[action_indices[self.ACTION_FPS_INDEX]],
            self.work_load_actions[action_indices[self.ACTION_WORK_LOAD_INDEX]],
        ]

        # Execute actions
        success = self._perform_actions(actions, slo_status)
        return actions, success

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

    def _setup_generative_model(self):
        self.obs_dims = [
            self.num_resolution_states,
            self.num_fps_states,
            self.num_work_load_states,
            len(SloStatus),
            len(SloStatus)
        ]

        self.state_dims = [
            self.num_resolution_states,
            self.num_fps_states,
            self.num_work_load_states,
        ]

        A = self._construct_A_matrix()
        B = self._construct_B_matrix()
        C = self._construct_C_matrix()
        D = self._construct_D_matrix()

        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            num_controls=self.control_dims,
            policy_len=self.policy_length,
            control_fac_idx=[0, 1, 2], # Indices that indicate which hidden state factors are directly controllable
            inference_algo="VANILLA",
            action_selection="deterministic"
        )

    def _construct_A_matrix(self):
        A = utils.obj_array(len(self.obs_dims))

        # Initialize observation modalities
        for obs_idx, obs_dim in enumerate(self.obs_dims):
            A[obs_idx] = np.zeros((obs_dim, *self.state_dims))

        # Direct mappings for resolution, fps and workload
        for i in range(self.num_resolution_states):
            A[self.OBS_RESOLUTION_INDEX][i, i, :, :] = 1.0

        for i in range(self.num_fps_states):
            A[self.OBS_FPS_INDEX][i, :, i, :] = 1.0

        for i in range(self.num_work_load_states):
            A[self.OBS_WORK_LOAD_INDEX][i, :, :, i] = 1.0

        # SLO probability mappings
        for res in range(self.num_resolution_states):
            for fps in range(self.num_fps_states):
                for wl in range(self.num_work_load_states):
                    queue_probs = self.slo_manager.get_qsize_slo_state_probabilities()
                    A[self.OBS_QUEUE_SIZE_INDEX][:, res, fps, wl] = queue_probs

                    mem_probs = self.slo_manager.get_mem_slo_state_probabilities()
                    A[self.OBS_MEMORY_USAGE_INDEX][:, res, fps, wl] = mem_probs

        return A

    def _construct_B_matrix(self):
        B = utils.obj_array(len(self.state_dims))

        # Create deterministic transition matrices
        B[0] = self._create_deterministic_transitions(self.num_resolution_states, len(self.resolution_actions))
        B[1] = self._create_deterministic_transitions(self.num_fps_states, len(self.fps_actions))
        B[2] = self._create_deterministic_transitions(self.num_work_load_states, len(self.work_load_actions))

        return B

    def _create_deterministic_transitions(self, num_states, num_actions):
        B = np.zeros((num_states, num_states, num_actions))

        for action in range(num_actions):
            for state in range(num_states):
                if action == ActionType.INCREASE.value:
                    next_state = min(state + 1, num_states - 1)
                elif action == ActionType.DECREASE.value:
                    next_state = max(state - 1, 0)
                else:
                    next_state = state

                B[next_state, state, action] = 1.0

        return B

    def _construct_C_matrix(self):
        C = utils.obj_array(len(self.obs_dims))

        # Initialize preferences
        for obs_idx, obs_dim in enumerate(self.obs_dims):
            C[obs_idx] = np.zeros(obs_dim)

        # Quality preferences (linear scaling)
        max_res = self.num_resolution_states - 1
        C[self.OBS_RESOLUTION_INDEX][:] = [self.MEDIUM_PREFERENCE * (i / max_res) for i in
                                           range(self.num_resolution_states)]

        max_fps = self.num_fps_states - 1
        C[self.OBS_FPS_INDEX][:] = [self.MEDIUM_PREFERENCE * (i / max_fps) for i in range(self.num_fps_states)]

        max_wl = self.num_work_load_states - 1
        C[self.OBS_WORK_LOAD_INDEX][:] = [self.LOW_PREFERENCE * (i / max_wl) for i in range(self.num_work_load_states)]

        # SLO preferences (strong aversion for critical states)
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.CRITICAL.value] = self.SLO_CRITICAL_AVERSION
        C[self.OBS_QUEUE_SIZE_INDEX][SloStatus.WARNING.value] = self.SLO_WARNING_AVERSION

        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.CRITICAL.value] = self.SLO_CRITICAL_AVERSION
        C[self.OBS_MEMORY_USAGE_INDEX][SloStatus.WARNING.value] = self.SLO_WARNING_AVERSION

        return C

    def _construct_D_matrix(self):
        D = utils.obj_array(len(self.state_dims))

        current_states = [
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index
        ]

        for i, state in enumerate(current_states):
            D[i] = np.zeros(self.state_dims[i])
            D[i][state] = 1.0

        return D

    def _get_slo_status(self, observations):
        return (
            SloStatus(observations[self.OBS_QUEUE_SIZE_INDEX]),
            SloStatus(observations[self.OBS_MEMORY_USAGE_INDEX])
        )

    def _adjust_actions_based_on_slo(self, action_indices, slo_status):
        queue_status, memory_status = slo_status

        actions = [
            self.resolution_actions[action_indices[self.ACTION_RESOLUTION_INDEX]],
            self.fps_actions[action_indices[self.ACTION_FPS_INDEX]],
            self.work_load_actions[action_indices[self.ACTION_WORK_LOAD_INDEX]],
        ]

        # Force decrease actions if any SLO is critical
        if queue_status == SloStatus.CRITICAL or memory_status == SloStatus.CRITICAL:
            return [
                ActionType.DECREASE if actions[0] == ActionType.INCREASE else actions[0],
                ActionType.DECREASE,
                ActionType.DECREASE
            ]

        # Moderate adjustment for warnings
        if queue_status == SloStatus.WARNING:
            actions[1] = ActionType.DECREASE

        if memory_status == SloStatus.WARNING:
            actions[0] = ActionType.DECREASE if actions[0] == ActionType.INCREASE else actions[0]

        return actions

    def _perform_actions(self, actions, slo_status):
        success = True

        # Resolution action
        if actions[self.ACTION_RESOLUTION_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_resolution()
        elif actions[self.ACTION_RESOLUTION_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_resolution()

        # FPS action
        if actions[self.ACTION_FPS_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_fps()
        elif actions[self.ACTION_FPS_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_fps()

        # Workload action
        if actions[self.ACTION_WORK_LOAD_INDEX] == ActionType.INCREASE:
            success &= self.elasticity_handler.increase_work_load()
        elif actions[self.ACTION_WORK_LOAD_INDEX] == ActionType.DECREASE:
            success &= self.elasticity_handler.decrease_work_load()

        return success

    def _get_slo_probabilities(self, res_idx, fps_idx, wl_idx, slo_type):
        """Calculate SLO state probabilities based on current configuration"""
        # Implementation should use your existing SLO management logic
        # This is a simplified example
        if slo_type == 'queue':
            load_factor = (res_idx + fps_idx + wl_idx) / 3
        else:  # memory
            load_factor = (res_idx * 2 + wl_idx) / 3

        if load_factor > 0.8:
            return [0.1, 0.2, 0.7]  # High probability of critical
        elif load_factor > 0.6:
            return [0.3, 0.5, 0.2]
        else:
            return [0.7, 0.2, 0.1]