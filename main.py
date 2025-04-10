from measurement.measurement import Measurement
from measurement.simulation.basic_simulation import BasicSimulation
from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from producer.global_variables import ProducerGlobalVariables
from worker.worker import Worker
from worker.worker_config import WorkerConfig


def create_workers(num: int, producer_ip: str, producer_port: int, collector_ip: str, collector_port: int):
    return [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port)) for i in range(num)]

def main():
    Measurement.run_all_simulations()


if __name__ == "__main__":
    main()
