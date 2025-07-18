from dataclasses import dataclass


@dataclass
class OutageConfig:
    time_until_outage: float
    duration: float



