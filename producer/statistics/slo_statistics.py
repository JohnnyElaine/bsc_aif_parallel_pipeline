import pandas as pd
from dataclasses import dataclass, field

@dataclass
class SloStatistics:
    queue_slo_satisfaction:list[bool] = field(default_factory=list)
    memory_slo_satisfaction:list[bool] = field(default_factory=list)
    queue_sizes:list[int] = field(default_factory=list)
    memory_usage:list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'queue_slo_satisfaction': self.queue_slo_satisfaction,
            'memory_slo_satisfaction': self.memory_slo_satisfaction,
            'queue_sizes': self.queue_sizes,
            'memory_usage': self.memory_usage
        })