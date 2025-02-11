from dataclasses import dataclass

from worker.enums.loading_mode import LoadingMode


@dataclass
class WorkerConfig:
    identity: int
    producer_ip: str
    producer_port: int



