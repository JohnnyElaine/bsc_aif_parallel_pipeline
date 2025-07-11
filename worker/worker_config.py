from dataclasses import dataclass

from worker.data.outage_config import OutageConfig


@dataclass
class WorkerConfig:
    identity: int
    producer_ip: str
    producer_port: int
    collector_ip: str
    collector_port: int
    processing_capacity: float
    outage_config: OutageConfig = None



