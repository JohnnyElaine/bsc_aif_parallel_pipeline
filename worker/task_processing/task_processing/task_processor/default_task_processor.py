from worker.task_processing.task_processing.task_processor.task_processor import TaskProcessor

class DefaultTaskProcessor(TaskProcessor):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def process(self, img):
        return img

    def change_work_load(self, img):
        pass