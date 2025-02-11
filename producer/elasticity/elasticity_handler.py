from packages.enums import WorkLoad
from producer.communication.request_handler import RequestHandler
from producer.data.resolution import Resolution
from producer.data.task_config import TaskConfig
from producer.task_generation.task_generator import TaskGenerator


class ElasticityHandler:
    def __init__(self, task_config: TaskConfig ,task_generator: TaskGenerator, request_handler: RequestHandler):
        self.config = task_config
        self._task_generator = task_generator
        self._request_handler = request_handler
        # TODO check if functions needs synchronization

    def change_work_load(self, work_load: WorkLoad):
        self.config.work_load = work_load
        self._request_handler.change_work_load(work_load)

    def change_fps(self, fps: int):
        self.config.fps = fps
        self._task_generator.set_fps(self.config.fps)

    def change_resolution(self, width: int, height: int):
        self.config.resolution = Resolution(width, height)
        self._task_generator.set_resolution(self.config.resolution)