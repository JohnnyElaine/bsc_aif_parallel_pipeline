import numpy as np
from pymdp import utils
from pymdp.agent import Agent

# Define the possible states for resolution, FPS, and workload
resolution_states = elasticity_handler._state_resolution.possible_states
fps_states = elasticity_handler._state_fps.possible_states
workload_states = elasticity_handler._state_work_load.possible_states

# Define the observation space (e.g., TPS, task queue size, current resolution)
observation_space = {
    'tps': np.array([0, 1]),  # Binary observation: below/above target TPS
    'task_queue': np.array([0, 1]),  # Binary observation: below/above target queue size
    'resolution': np.array([0, 1])  # Binary observation: below/above target resolution
}

# Define the action space
action_space = {
    'resolution': ['increase_resolution', 'decrease_resolution'],
    'fps': ['increase_fps', 'decrease_fps'],
    'workload': ['increase_work_load', 'decrease_work_load']
}

# Define the transition model (how hidden states change over time)
num_res_states = len(resolution_states)
num_fps_states = len(fps_states)
num_workload_states = len(workload_states)

transition_matrix_resolution = utils.random_A_matrix(num_res_states, num_res_states)
transition_matrix_fps = utils.random_A_matrix(num_fps_states, num_fps_states)
transition_matrix_workload = utils.random_A_matrix(num_workload_states, num_workload_states)

# Define the likelihood model (how observations are generated from hidden states)
likelihood_matrix = utils.random_A_matrix(len(observation_space['tps']), num_res_states * num_fps_states * num_workload_states)

# Define the prior over initial states (D)
D = utils.onehot(0, num_res_states * num_fps_states * num_workload_states)  # Assume the system starts in the first state, TODO: set correct starting state

# Define the prior over policies (E)
E = utils.onehot(0, len(action_space['resolution']) * len(action_space['fps']) * len(action_space['workload']))  # Uniform prior over policies

# Create the active inference agent
agent = Agent(
    A=likelihood_matrix,  # Likelihood model
    B=[transition_matrix_resolution, transition_matrix_fps, transition_matrix_workload],  # Transition models
    C=utils.onehot(0, len(observation_space['tps'])),  # Prior preferences (prefer high TPS, low queue size, high resolution)
    D=D,  # Prior over initial states
    E=E,  # Prior over policies
    policy_len=1  # Policy horizon (1 step lookahead)
)


def observe_system(self):
    """
    Observe the current state of the system and return an observation vector.
    """
    current_tps = self.get_current_tps()
    task_queue_size = self.get_task_queue_size()
    current_res = self.get_current_resolution()

    # Convert observations to a binary vector
    observation = {
        'tps': 0 if current_tps < self.source_fps * self.tolerance else 1,
        'task_queue': 0 if task_queue_size <= self.source_fps * 2 else 1,
        'resolution': 0 if current_res < self.source_res * self.tolerance else 1
    }

    return observation


def step(self):
    """
    Perform a single step of the active inference loop.
    """
    # Observe the system
    observation = self.observe_system()

    # Update the agent's beliefs
    self.agent.infer_states(observation)

    # Decide on actions
    action = self.agent.sample_action()

    # Execute the action
    self.execute_action(action)


def execute_action(self, action):
    """
    Execute the action using the ElasticityHandler.
    """
    if action == 'increase_resolution':
        self.elasticity_handler.increase_resolution()
    elif action == 'decrease_resolution':
        self.elasticity_handler.decrease_resolution()
    elif action == 'increase_fps':
        self.elasticity_handler.increase_fps()
    elif action == 'decrease_fps':
        self.elasticity_handler.decrease_fps()
    elif action == 'increase_work_load':
        self.elasticity_handler.increase_work_load()
    elif action == 'decrease_work_load':
        self.elasticity_handler.decrease_work_load()