from multiprocessing import Manager

import pandas as pd

from collector.collector import Collector
from collector.collector_config import CollectorConfig
from evaluation.simulation.simulation import Simulation
from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.enums.agent_type import AgentType
from producer.producer import Producer
from producer.producer_config import ProducerConfig
from worker.worker import Worker
from worker.worker_config import WorkerConfig


class BaseCaseSimulation(Simulation):
    """
    All workers start at the same time and stop when the producer is finished
    """

    def __init__(self, producer_ip: str, producer_port: int, collector_ip: str, collector_port: int,
                 work_type: WorkType, loading_mode: LoadingMode, max_inference_quality: InferenceQuality, agent_type: AgentType,
                 vid_path: str, worker_capacities: list[float]):
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_inference_quality, agent_type, vid_path)
        self.worker_capacities = worker_capacities

    def run(self) -> dict[str, pd.DataFrame]:
        producer_config = ProducerConfig(
            port=self.producer_port,
            work_type=self.work_type,
            loading_mode=self.loading_mode,
            max_inference_quality=self.max_inference_quality,
            agent_type=self.agent_type,
            video_path=self.vid_path,
            track_slo_stats=True,
            initial_stream_multiplier=1
        )
        collector_config = CollectorConfig(self.collector_port)

        stats = None

        with Manager() as manager:
            stats_multiprocess = manager.dict()

            producer = Producer(producer_config, shared_stats_dict=stats_multiprocess)
            workers = self.create_workers(self.worker_capacities, self.producer_ip,
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

            stats = dict(stats_multiprocess)

        return stats

    def create_workers(self, capacities: list[float], producer_ip: str, producer_port: int, collector_ip: str,
                       collector_port: int):
        return [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port, capacity)) for
                i, capacity
                in enumerate(capacities)]

