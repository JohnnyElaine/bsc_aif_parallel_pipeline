import pandas as pd
from multiprocessing import Manager

from collector.collector import Collector
from collector.collector_config import CollectorConfig
from measurement.simulation.simulation import Simulation
from packages.data import Video

from packages.enums import WorkType, LoadingMode, WorkLoad
from producer.enums.agent_type import AgentType
from producer.producer import Producer
from producer.producer_config import ProducerConfig
from worker.data.outage_config import OutageConfig
from worker.worker import Worker
from worker.worker_config import WorkerConfig


class OutageAndRecoverySimulation(Simulation):
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
                 max_work_load: WorkLoad,
                 agent_type: AgentType,
                 worker_capacities: list[float],
                 vid_path: str,
                 outage_at: float,
                 recovery_at):
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_work_load, agent_type, worker_capacities, vid_path)
        self.outage_at = outage_at
        self.recovery_at = recovery_at


    def run(self) -> dict[str, pd.DataFrame]:
        producer_config = ProducerConfig(self.producer_port, self.work_type,
                                         self.loading_mode, self.max_work_load,
                                         self.agent_type, self.vid_path)
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

    def create_workers(self, capacities: list[float], producer_ip: str, producer_port: int, collector_ip: str, collector_port: int):
        video = Video(self.vid_path)
        fps = video.fps
        total_num_of_frames = video.frame_count
        total_capacity = sum(capacities)

        workers = []

        for i, capacity in enumerate(capacities):
            config = WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port, capacity)
            outage_config = OutageAndRecoverySimulation.create_outage_config(capacity, total_num_of_frames,
                                                                             total_capacity, fps,
                                                                             self.outage_at,
                                                                             self.recovery_at)
            workers.append(Worker(config, outage_config=outage_config))

        return workers

    @staticmethod
    def create_outage_config(worker_capacity: float,
                             total_num_of_tasks: int,
                             total_capacity: float,
                             fps: int,
                             outage_at_perc: float,
                             recovery_at_perc: float) -> OutageConfig:

        expected_tasks = total_num_of_tasks * worker_capacity / total_capacity
        frames_until_outage = round(expected_tasks * outage_at_perc)
        frames_until_recovery = round(expected_tasks * recovery_at_perc)
        duration = (frames_until_recovery - frames_until_outage) / fps

        return OutageConfig(frames_until_outage, duration)

