from measurement.simulation_config import SimulationConfig
from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from producer.global_variables import ProducerGlobalVariables
from producer.producer_config import ProducerConfig
from producer.producer import Producer
from worker.global_variables import WorkerGlobalVariables
from worker.worker import Worker
from worker.worker_config import WorkerConfig
from collector.collector import Collector
from collector.collector_config import CollectorConfig

class Measurement:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'

    def __init__(self):
        pass

    def simulation1(self):
        vid_path = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
        num_workers = 2

        producer_config = ProducerConfig(10000, vid_path, WorkType.YOLO_DETECTION, WorkLoad.LOW, LoadingMode.LAZY)
        collector_config = CollectorConfig(10001)

        producer = Producer(producer_config)
        workers = Measurement.create_workers(num_workers, 'localhost', producer_config.port, 'localhost', collector_config.port)
        collector = Collector(collector_config)

        for worker in workers:
            worker.start()

        collector.start()
        producer.run()

    def run_all_simulations(self):
        vid_path = ProducerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
        num_workers = 3

        for agent_type in AgentType:
            config = SimulationConfig(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                                      Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, LoadingMode.EAGER, WorkLoad.HIGH,
                                      agent_type, num_workers, vid_path)
            self.run_basic_simulation(config)

    def run_basic_simulation(self, sim_config: SimulationConfig):
        producer_config = ProducerConfig(sim_config.producer_port, sim_config.work_type,
                                         sim_config.loading_mode, sim_config.max_work_load,
                                         sim_config.agent_type, sim_config.vid_path)
        collector_config = CollectorConfig(sim_config.collector_port)

        process_delay = 0

        producer = Producer(producer_config)
        workers = Measurement.create_workers(sim_config.num_workers, sim_config.producer_ip,
                                             sim_config.producer_port, sim_config.collector_ip,
                                             sim_config.collector_port, process_delay)
        collector = Collector(collector_config)

        collector.start()
        producer.start()


        for worker in workers:
            worker.start()

        # collect simulation data

        # wait for simulation to complete
        producer.join()
        for worker in workers:
            worker.join()
        collector.join()


    @staticmethod
    def create_workers(num: int, producer_ip: str, producer_port: int, collector_ip: str, collector_port: int, process_delay_s: float):
        return [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port, process_delay_s)) for i in range(num)]