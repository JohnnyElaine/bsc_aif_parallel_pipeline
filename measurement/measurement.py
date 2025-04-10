from measurement.simulation.basic_simulation import BasicSimulation
from packages.enums import WorkType, WorkLoad
from packages.enums.loading_mode import LoadingMode
from producer.enums.agent_type import AgentType
from producer.global_variables import ProducerGlobalVariables
from producer.producer_config import ProducerConfig
from producer.producer import Producer
from worker.global_variables import WorkerGlobalVariables
from collector.collector import Collector
from collector.collector_config import CollectorConfig

class Measurement:

    PRODUCER_PORT = 10000
    COLLECTOR_PORT = 10001
    LOCALHOST = 'localhost'
    VID_PATH = WorkerGlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'

    def __init__(self):
        pass

    def simulation(self):
        sim = BasicSimulation(Measurement.LOCALHOST, Measurement.PRODUCER_PORT, Measurement.LOCALHOST,
                              Measurement.COLLECTOR_PORT, WorkType.YOLO_DETECTION, LoadingMode.LAZY, WorkLoad.HIGH,
                              AgentType.ACTIVE_INFERENCE, 0, )

    def simulation1(self):
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

