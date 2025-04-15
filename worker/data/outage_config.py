from dataclasses import dataclass


@dataclass
class OutageConfig:
    frames_until_outage: int
    duration: float



