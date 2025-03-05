from pymdp import utils

from producer.elasticity.elasticity_handler import ElasticityHandler


class ActiveInferenceAgentBasic:
    def __init__(self, elasticity_handler: ElasticityHandler, target_fps, target_resolution, tolerance=0.9):
        self.elasticity_handler = elasticity_handler
        self.source_fps = target_fps
        self.source_res = target_resolution
        self.tolerance = tolerance

        # Define the possible states and actions
        self.states = {
            'resolution': elasticity_handler.state_resolution.possible_states,
            'fps': elasticity_handler.state_fps.possible_states,
            'work_load': elasticity_handler.state_work_load.possible_states
        }

        self.actions = {
            'resolution': ['increase_resolution', 'decrease_resolution'],
            'fps': ['increase_fps', 'decrease_fps'],
            'work_load': ['increase_work_load', 'decrease_work_load']
        }

        # Initialize the agent's beliefs about the states
        self.beliefs = {
            'resolution': utils.onehot(elasticity_handler.resolution, self.states['resolution']),
            'fps': utils.onehot(elasticity_handler.fps, self.states['fps']),
            'work_load': utils.onehot(elasticity_handler.work_load, self.states['work_load'])
        }

    def update_beliefs(self, current_tps, task_queue_size, current_res):
        """
        Update the agent's beliefs based on the current system state.
        """
        # Update beliefs based on the current state
        self.beliefs['resolution'] = utils.onehot(current_res, self.states['resolution'])
        self.beliefs['fps'] = utils.onehot(current_tps, self.states['fps'])
        self.beliefs['work_load'] = utils.onehot(task_queue_size, self.states['work_load'])

    def decide_action(self):
        """
        Decide which action to take based on the current beliefs and SLOs.
        """
        # Check SLOs and decide actions
        actions = []

        # Task queue size SLO
        if task_queue_size > self.source_fps * 2:
            actions.append('decrease_work_load')

        # FPS SLO
        if current_fps < self.source_fps * self.tolerance:
            actions.append('increase_fps')

        # Resolution SLO
        if current_res < self.source_res * self.tolerance:
            actions.append('increase_resolution')

        # Quality/Workload SLO
        if current_tps >= self.source_fps * self.tolerance and task_queue_size <= self.source_fps * 2:
            actions.append('increase_work_load')

        return actions

    def execute_actions(self, actions):
        """
        Execute the decided actions using the ElasticityHandler.
        """
        for action in actions:
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

    def step(self, current_tps, task_queue_size, current_res):
        """
        Perform a single step of the active inference loop.
        """
        self.update_beliefs(current_tps, task_queue_size, current_res)
        actions = self.decide_action()
        self.execute_actions(actions)