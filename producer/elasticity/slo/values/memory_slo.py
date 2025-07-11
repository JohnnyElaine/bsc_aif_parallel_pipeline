import psutil

from producer.elasticity.slo.slo_util import SloUtil


class MemorySlo:
    def __init__(self, max_mem_usage_float_percent, stats=None):
        self._max_mem_usage_float_percent = max_mem_usage_float_percent
        self._stats = stats

    def value(self, track_stats=True):
        mem_usage = psutil.virtual_memory().percent / 100
        value = mem_usage / self._max_mem_usage_float_percent

        if track_stats and (self._stats is not None):
            self._stats.memory_usage.append(mem_usage)
            self._stats.memory_usage_slo_value.append(value)

        return value

    def probabilities(self) -> list:
        """
        Calculate probabilities for each memory SLO state (OK, WARNING, CRITICAL).
        Uses the value of current memory usage to maximum allowed memory usage.

        Returns:
            list: Probabilities for each SLO state [p_ok, p_warning, p_critical]
        """
        return SloUtil.get_slo_state_probabilities(self.value(track_stats=False))
