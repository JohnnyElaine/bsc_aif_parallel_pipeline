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
    vid_path = ProducerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'

    worker_process_delays_s =  [0, 0, 0]

    simulations = []

    for agent_type in AgentType:
        s1 = BasicSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                                  Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, LoadingMode.EAGER, WorkLoad.HIGH,
                                  agent_type, worker_process_delays_s, vid_path)

        simulations.append(s1)


    simulations[0].run()


if __name__ == "__main__":
    main()
