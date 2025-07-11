import pandas as pd
from multiprocessing import Manager

from collector.collector import Collector
from collector.collector_config import CollectorConfig
from evaluation.simulation.simulation import Simulation
from packages.data import Video

from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.enums.agent_type import AgentType
from producer.producer import Producer
from producer.producer_config import ProducerConfig
from worker.data.outage_config import OutageConfig
from worker.worker import Worker
from worker.worker_config import WorkerConfig


class VariableComputationalBudgetSimulation(Simulation):
    """
    All workers start at the same time and stop when the producer is finished
    Pre-defined set of workers stop requesting tasks at a given % (outage_at) and
    resume at a given % (recovery_at) of the total simulation time
    """

    def __init__(self,
                 producer_ip: str,
                 producer_port: int,
                 collector_ip: str,
                 collector_port: int,
                 work_type: WorkType,
                 loading_mode: LoadingMode,
                 max_work_load: InferenceQuality,
                 agent_type: AgentType,
                 vid_path: str,
                 regular_worker_capacities: list[float],
                 outage_worker_capacities: list[float],
                 outage_at: float,
                 recovery_at: float):
        """
        Total number of workers = len(regular_worker_capacities) + len(outage_worker_capacities)
        Args:
            producer_ip:
            producer_port:
            collector_ip:
            collector_port:
            work_type:
            loading_mode:
            max_work_load:
            agent_type:
            vid_path:
            regular_worker_capacities:
            outage_worker_capacities:
            outage_at:
            recovery_at:
        """
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_work_load, agent_type, vid_path)

        self._regular_worker_capacities = regular_worker_capacities
        self._outage_worker_capacities = outage_worker_capacities
        self._outage_at = outage_at
        self._recovery_at = recovery_at

    def run(self) -> dict[str, pd.DataFrame]:
        producer_config = ProducerConfig(self.producer_port, self.work_type,
                                         self.loading_mode, self.max_work_load,
                                         self.agent_type, self.vid_path)
        collector_config = CollectorConfig(self.collector_port)

        stats = None

        with Manager() as manager:
            stats_multiprocess = manager.dict()

            producer = Producer(producer_config, shared_stats_dict=stats_multiprocess)
            workers = self.create_workers(self._regular_worker_capacities, self._outage_worker_capacities,
                                          self.producer_ip,self.producer_port, self.collector_ip, self.collector_port)
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

    def create_workers(self, regular_worker_capacities: list[float], outage_worker_capacities: list[float] ,producer_ip: str, producer_port: int, collector_ip: str, collector_port: int):
        video = Video(self.vid_path)
        fps = video.fps
        total_num_of_frames = video.frame_count
        total_capacity = sum(regular_worker_capacities + outage_worker_capacities)

        # regular workers
        workers = [Worker(WorkerConfig(i, producer_ip, producer_port, collector_ip, collector_port, capacity)) for i, capacity in enumerate(regular_worker_capacities)]

        # outage workers, i.e. worker that will temporarily halt task processing for a set amount of time to simulate an outage
        for i, capacity in enumerate(outage_worker_capacities):
            identity = i + len(regular_worker_capacities)
            outage_config = VariableComputationalBudgetSimulation.create_outage_config(capacity, total_num_of_frames,
                                                                                       total_capacity, fps,
                                                                                       self._outage_at,
                                                                                       self._recovery_at)
            config = WorkerConfig(identity, producer_ip, producer_port, collector_ip, collector_port, capacity, outage_config=outage_config)
            workers.append(Worker(config))

        return workers

    @staticmethod
    def create_outage_config(worker_capacity: float,
                             total_num_of_tasks: int,
                             total_capacity: float,
                             fps: int,
                             outage_at_perc: float,
                             recovery_at_perc: float) -> OutageConfig:
        frames_until_outage = round(total_num_of_tasks * worker_capacity / total_capacity * outage_at_perc)
        duration = (total_num_of_tasks * recovery_at_perc - total_num_of_tasks * outage_at_perc) / fps

        return OutageConfig(frames_until_outage, duration)

