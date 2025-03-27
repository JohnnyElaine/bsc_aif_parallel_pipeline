from dataclasses import dataclass, field

@dataclass
class SloStatistics:
    queue_slo_satisfaction:list[bool] = field(default_factory=list)
    memory_slo_satisfaction:list[bool] = field(default_factory=list)
    queue_sizes:list[int] = field(default_factory=list)
    memory_usage:list[float] = field(default_factory=list)