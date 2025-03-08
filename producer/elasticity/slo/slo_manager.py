import psutil
from producer.data.resolution import Resolution
from producer.elasticity.elasticity_handler import ElasticityHandler


class SloManager:
    def __init__(self, elasticity_handler: ElasticityHandler,
                 target_fps: int,
                 target_resolution: Resolution,
                 max_memory_percentage = 80):
        self.elasticity_handler = elasticity_handler

        self.target_fps = target_fps
        self.target_resolution = target_resolution

        self.max_memory_percentage = max_memory_percentage

    def queue_slo_satisfied(self) -> bool:
        max_qsize = 2 * self.elasticity_handler.fps # Example: 2 seconds worth of frames
        return self.elasticity_handler.queue_size() <= max_qsize

    def memory_slo_satisfied(self) -> bool:
        return psutil.virtual_memory().percent <= self.max_memory_percentage