import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from producer.elasticity.agent.action.action_type import ActionType
from producer.elasticity.agent.elasticity_agent import ElasticityAgent
from producer.elasticity.slo.slo_status import SloStatus
from producer.elasticity.handler.elasticity_handler import ElasticityHandler

# Set up logging
log = logging.getLogger('producer')


class VideoProcessorEnv(gym.Env):
    """
    Custom Environment for video processing system that follows gym interface.
    This environment models the state of a distributed video processing system
    and handles the reinforcement learning state, actions, and rewards.
    """
    metadata = {'render.modes': ['human']}

    # Reward constants
    SLO_CRITICAL_PENALTY = -10.0
    SLO_WARNING_PENALTY = -2.0
    SLO_OK_REWARD = 2.0

    QUALITY_REWARD_FACTOR = 0.5  # Factor for quality improvements

    def __init__(self, elasticity_handler: ElasticityHandler, slo_manager):
        super().__init__()

        self.elasticity_handler = elasticity_handler
        self.slo_manager = slo_manager

        # Define action space
        self.action_space = spaces.Discrete(len(ActionType))

        # Define observation space
        # [resolution_idx, fps_idx, workload_idx, queue_size_status, memory_usage_status]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([
                len(elasticity_handler.state_resolution.possible_states) - 1,
                len(elasticity_handler.state_fps.possible_states) - 1,
                len(elasticity_handler.state_work_load.possible_states) - 1,
                len(SloStatus) - 1,
                len(SloStatus) - 1
            ]),
            dtype=np.int32
        )

        # Track previous state for reward calculation
        self.prev_resolution_idx = elasticity_handler.state_resolution.current_index
        self.prev_fps_idx = elasticity_handler.state_fps.current_index
        self.prev_workload_idx = elasticity_handler.state_work_load.current_index

    def reset(self):
        """Reset the environment to the current system state"""
        # Get the current state
        observation = self._get_observation()

        # Update previous state tracking
        self.prev_resolution_idx = self.elasticity_handler.state_resolution.current_index
        self.prev_fps_idx = self.elasticity_handler.state_fps.current_index
        self.prev_workload_idx = self.elasticity_handler.state_work_load.current_index

        return observation

    def step(self, action):
        """
        Take action in the environment

        Args:
            action: Integer representing the action to take

        Returns:
            observation, reward, done, info
        """
        action_type = ActionType(action)
        success = self._perform_action(action_type)

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(success)

        # Update previous state tracking
        self.prev_resolution_idx = self.elasticity_handler.state_resolution.current_index
        self.prev_fps_idx = self.elasticity_handler.state_fps.current_index
        self.prev_workload_idx = self.elasticity_handler.state_work_load.current_index

        # We don't have an episode termination condition in this continuous task
        done = False

        # Additional info for debugging
        info = {
            'action_success': success,
            'action': action_type.name,
            'resolution': str(self.elasticity_handler.resolution),
            'fps': self.elasticity_handler.fps,
            'workload': self.elasticity_handler.work_load.name,
            'queue_status': self.slo_manager.get_qsize_slo_status().name,
            'memory_status': self.slo_manager.get_memory_slo_status().name
        }

        return observation, reward, done, info

    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            queue_status = self.slo_manager.get_qsize_slo_status()
            memory_status = self.slo_manager.get_memory_slo_status()

            status_str = (
                f"Resolution: {self.elasticity_handler.resolution}, "
                f"FPS: {self.elasticity_handler.fps}, "
                f"WorkLoad: {self.elasticity_handler.work_load.name}, "
                f"Queue Status: {queue_status.name}, "
                f"Memory Status: {memory_status.name}"
            )
            print(status_str)
        return None

    def _get_observation(self):
        """Get the current state of the system as an observation"""
        return np.array([
            self.elasticity_handler.state_resolution.current_index,
            self.elasticity_handler.state_fps.current_index,
            self.elasticity_handler.state_work_load.current_index,
            self.slo_manager.get_qsize_slo_status().value,
            self.slo_manager.get_memory_slo_status().value
        ], dtype=np.int32)

    def _perform_action(self, action: ActionType) -> bool:
        """
        Execute the specified action on the elasticity handler

        Args:
            action: The action to perform

        Returns:
            bool: Whether the action was successful
        """
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
                log.warning(f"Unknown action: {action}")
                return False

    def _calculate_reward(self, action_success: bool) -> float:
        """
        Calculate the reward based on SLO status and quality improvements

        Args:
            action_success: Whether the action was successful

        Returns:
            float: The calculated reward
        """
        if not action_success:
            # Small penalty for unsuccessful actions
            return -0.5

        reward = 0.0

        # SLO rewards/penalties - these have highest priority
        queue_status = self.slo_manager.get_qsize_slo_status()
        memory_status = self.slo_manager.get_memory_slo_status()

        # Calculate SLO rewards
        for status in [queue_status, memory_status]:
            if status == SloStatus.CRITICAL:
                reward += self.SLO_CRITICAL_PENALTY
            elif status == SloStatus.WARNING:
                reward += self.SLO_WARNING_PENALTY
            else:  # SloStatus.OK
                reward += self.SLO_OK_REWARD

        # Quality rewards - secondary priority
        # Only give quality rewards if SLO status is not critical
        if queue_status != SloStatus.CRITICAL and memory_status != SloStatus.CRITICAL:
            # Resolution improvement
            resolution_diff = self.elasticity_handler.state_resolution.current_index - self.prev_resolution_idx
            reward += resolution_diff * self.QUALITY_REWARD_FACTOR

            # FPS improvement
            fps_diff = self.elasticity_handler.state_fps.current_index - self.prev_fps_idx
            reward += fps_diff * self.QUALITY_REWARD_FACTOR

            # Workload improvement
            workload_diff = self.elasticity_handler.state_work_load.current_index - self.prev_workload_idx
            reward += workload_diff * self.QUALITY_REWARD_FACTOR

        return reward


class ReinforcementLearningAgent(ElasticityAgent):
    """
    Reinforcement Learning Agent that uses Stable Baselines3 to maintain
    Service Level Objectives (SLOs) in a distributed video processing system.

    The agent monitors the system's state and takes actions to optimize the quality
    of experience while ensuring computational resources are properly utilized.
    """

    def __init__(self, elasticity_handler: ElasticityHandler, model_path: str = None):
        """
        Initialize the Reinforcement Learning Agent

        Args:
            elasticity_handler: The handler for changing system parameters
            model_path: Path to a pre-trained model to load (optional)
        """
        super().__init__(elasticity_handler)

        # Create the environment
        self.env = VideoProcessorEnv(elasticity_handler, self.slo_manager)

        # Wrap the environment for logging and normalization
        self.env = Monitor(self.env)
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)

        # Create or load the agent
        if model_path:
            log.info(f"Loading pre-trained model from {model_path}")
            self.model = PPO.load(model_path, env=self.env)
        else:
            log.info("Creating new PPO model")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./tensorboard_logs/"
            )

        # Reset the environment to the current state
        self.obs = self.env.reset()

        # Store history for analysis
        self.action_history = []
        self.reward_history = []
        self.state_history = []

    def step(self) -> Tuple[ActionType, bool]:
        """
        Perform a single step of reinforcement learning

        Returns:
            tuple[ActionType, bool]: The action taken and whether it was successful
        """
        # Get the action from the model
        action, _ = self.model.predict(self.obs, deterministic=True)

        # Take the action in the environment
        self.obs, reward, _, info = self.env.step(action)

        # Store history
        action_type = ActionType(action[0])
        action_success = info[0]['action_success']

        self.action_history.append(action_type)
        self.reward_history.append(reward[0])
        self.state_history.append(self._get_current_state_dict())

        log.info(f"Action: {action_type.name}, Success: {action_success}, Reward: {reward[0]:.2f}")

        return action_type, action_success

    def train(self, total_timesteps: int = 10000):
        """
        Train the reinforcement learning model

        Args:
            total_timesteps: Total number of timesteps for training
        """
        log.info(f"Training model for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps)
        log.info("Training completed")

    def save(self, path: str):
        """
        Save the model to a file

        Args:
            path: Path to save the model
        """
        self.model.save(path)
        log.info(f"Model saved to {path}")

        # Also save the normalization stats
        self.env.save(f"{path}_vecnormalize.pkl")

    def _get_current_state_dict(self) -> Dict:
        """Get the current state as a dictionary for history tracking"""
        return {
            'resolution': str(self.elasticity_handler.resolution),
            'fps': self.elasticity_handler.fps,
            'workload': self.elasticity_handler.work_load.name,
            'queue_status': self.slo_manager.get_qsize_slo_status().name,
            'memory_status': self.slo_manager.get_memory_slo_status().name
        }

    def get_action_statistics(self) -> pd.DataFrame:
        """
        Get statistics about actions taken

        Returns:
            pd.DataFrame: Statistics about actions taken
        """
        if not self.action_history:
            return pd.DataFrame()

        action_counts = {}
        for action in ActionType:
            action_counts[action.name] = self.action_history.count(action)

        return pd.DataFrame(action_counts.items(), columns=['Action', 'Count'])

    def get_performance_history(self) -> pd.DataFrame:
        """
        Get the history of states and rewards

        Returns:
            pd.DataFrame: Performance history
        """
        if not self.state_history:
            return pd.DataFrame()

        history_df = pd.DataFrame(self.state_history)
        history_df['reward'] = self.reward_history
        history_df['action'] = [action.name for action in self.action_history]

        return history_df