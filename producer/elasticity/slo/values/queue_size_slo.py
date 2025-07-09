from producer.elasticity.slo.slo_util import SloUtil
from producer.task_generation.task_generator import TaskGenerator

class QueueSizeSlo:

    def __init__(self, task_generator: TaskGenerator, tolerance=1, stats=None):
        self._task_generator = task_generator
        self._tolerance = tolerance
        self._stats = stats

    def value(self, track_stats=True):
        queue_size = self._task_generator.queue_size()
        max_queue_size = self._task_generator.fps * self._tolerance
        value = queue_size / max_queue_size

        if track_stats and (self._stats is not None):
            self._stats.queue_size.append(queue_size)
            self._stats.queue_size_slo_value.append(value)

        return value

    def probabilities(self) -> list:
        """
        Calculate probabilities for each queue SLO state (OK, WARNING, CRITICAL).
        Uses the value of current queue size to maximum allowed queue size.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloUtil.get_slo_state_probabilities(self.value(track_stats=False))