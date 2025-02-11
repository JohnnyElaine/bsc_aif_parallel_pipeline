from dataclasses import dataclass

@dataclass(frozen=True)
class CollectorConfig:
    port: int
