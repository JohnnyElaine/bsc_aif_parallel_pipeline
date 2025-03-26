import psutil
from producer.elasticity.handler.elasticity_handler import ElasticityHandler


class SloManager:
    def __init__(self, elasticity_handler: ElasticityHandler,
                 max_memory_percentage = 80):
        self.elasticity_handler = elasticity_handler
        self.max_memory_percentage = max_memory_percentage

    def is_queue_slo_satisfied(self) -> bool:
        return self.elasticity_handler.queue_size() <= self._max_qsize()

    def is_memory_slo_satisfied(self) -> bool:
        return psutil.virtual_memory().percent <= self.max_memory_percentage

    def probability_queue_slo_satisfied(self) -> float:
        return 1 - self.elasticity_handler.queue_size() / self._max_qsize()

    def probability_memory_slo_satisfied(self) -> float:
        return 1 - psutil.virtual_memory().percent / self.max_memory_percentage

    def _max_qsize(self):
        return 2 * self.elasticity_handler.fps # Example: 2 seconds worth of frames