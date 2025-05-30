import pandas as pd
from multiprocessing import Manager

from collector.collector import Collector
from collector.collector_config import CollectorConfig
from measurement.simulation.simulation import Simulation

from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.enums.agent_type import AgentType
from producer.producer import Producer
from producer.producer_config import ProducerConfig


class DelayedStartSimulation(Simulation):
    """
    All workers start at the same time and stop when the producer is finished
    """

    def __init__(self,
                 producer_ip: str,
                 producer_port: int,
                 collector_ip: str,
                 collector_port: int,
                 work_type: WorkType,
                 loading_mode: LoadingMode,
                 max_inference_quality: InferenceQuality,
                 agent_type: AgentType,
                 worker_capacities: list[float],
                 vid_path: str):
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_inference_quality, agent_type, worker_capacities, vid_path)

    def run(self) -> dict[str, pd.DataFrame]:
        producer_config = ProducerConfig(self.producer_port, self.work_type,
                                         self.loading_mode, self.max_work_load,
                                         self.agent_type, self.vid_path)
        collector_config = CollectorConfig(self.collector_port)

        stats = None

        with Manager() as manager:
            stats = manager.dict()

            producer = Producer(producer_config, stats)
            workers = Simulation.create_workers(self.worker_capacities, self.producer_ip,
                                                self.producer_port, self.collector_ip,
                                                self.collector_port)
            collector = Collector(collector_config)

            collector.start()
            producer.start()

            for worker in workers:
                worker.start()

            # wait for simulation to complete
            producer.join()
            for worker in workers:
                worker.join()
            collector.join()

            # collect simulation data
            #stats = producer.get_statistics()

        return stats