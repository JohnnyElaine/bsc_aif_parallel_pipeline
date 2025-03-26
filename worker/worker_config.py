from dataclasses import dataclass


@dataclass
class WorkerConfig:
    identity: int
    producer_ip: str
    producer_port: int
    collector_ip: str
    collector_port: int
    process_delay_s: float



