from dataclasses import dataclass

from worker.enums.loading_mode import LoadingMode


@dataclass
class WorkerConfig:
    identity: int
    model_loading_mode: LoadingMode
    producer_ip: str
    producer_port: int
    is_simulation: bool



