from multiprocessing import Manager

import pandas as pd

from collector.collector import Collector
from collector.collector_config import CollectorConfig
from evaluation.simulation.simulation import Simulation
from packages.enums import WorkType, LoadingMode, InferenceQuality
from producer.data.stream_multiplier_entry import StreamMultiplierEntry
from producer.enums.agent_type import AgentType
from producer.producer import Producer
from producer.producer_config import ProducerConfig
from worker.worker import Worker
from worker.worker_config import WorkerConfig


class VariableComputationalDemandSimulation(Simulation):
    """
    Simulation that varies computational demand by changing the stream multiplier at specific points
    in the video processing timeline. This simulates multiple streams being processed simultaneously.
    
    The stream multiplier changes are handled automatically within the producer's elasticity system.
    
    Timeline:
    - 0-25%: Single stream (multiplier=1)
    - 25-75%: Double stream (multiplier=2) 
    - 75-100%: Single stream (multiplier=1)
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
                 vid_path: str,
                 worker_capacities: list[float],
                 stream_multiplier_increase_at,
                 stream_multiplier_decrease_at,
                 stream_multiplier_increase_to,
                 stream_multiplier_decrease_to):
        """
        Args:
            producer_ip: Producer IP address
            producer_port: Producer port
            collector_ip: Collector IP address  
            collector_port: Collector port
            work_type: Type of work to perform
            loading_mode: Loading mode for workers
            max_inference_quality: Maximum inference quality
            agent_type: Type of AIF agent to use
            vid_path: Path to video file
            worker_capacities: List of worker processing capacities
            stream_multiplier_increase_at: Percentage of frames processed when multiplier increases
            stream_multiplier_decrease_at: Percentage of frames processed when multiplier decreases
            stream_multiplier_increase_to: Maximum stream multiplier value
        """
        super().__init__(producer_ip, producer_port, collector_ip, collector_port, work_type, loading_mode,
                         max_inference_quality, agent_type, vid_path)
        
        self._worker_capacities = worker_capacities
        self._stream_multiplier_schedule = [
            StreamMultiplierEntry(stream_multiplier_increase_at, stream_multiplier_increase_to),
            StreamMultiplierEntry(stream_multiplier_decrease_at, stream_multiplier_decrease_to)
        ]

    def run(self) -> dict[str, pd.DataFrame]:
        producer_config = ProducerConfig(
            port=self.producer_port,
            work_type=self.work_type,
            loading_mode=self.loading_mode,
            max_inference_quality=self.max_inference_quality,
            agent_type=self.agent_type,
            video_path=self.vid_path,
            track_slo_stats=True,
            initial_stream_multiplier=1,  # Start with single stream
            stream_multiplier_schedule=self._stream_multiplier_schedule
        )
        
        collector_config = CollectorConfig(self.collector_port)
        
        stats = None
        
        with Manager() as manager:
            stats_multiprocess = manager.dict()
            
            producer = Producer(producer_config, shared_stats_dict=stats_multiprocess)
            workers = self._create_workers()
            collector = Collector(collector_config)
            
            # Start all components
            collector.start()
            producer.start()
            
            for worker in workers:
                worker.start()
            
            # Wait for simulation to complete
            producer.join()
            for worker in workers:
                worker.join()
            collector.join()
            
            stats = dict(stats_multiprocess)
        
        return stats

    def _create_workers(self) -> list[Worker]:
        """Create worker processes with specified capacities"""
        workers = []
        for i, capacity in enumerate(self._worker_capacities):
            config = WorkerConfig(
                identity=i,
                producer_ip=self.producer_ip,
                producer_port=self.producer_port,
                collector_ip=self.collector_ip,
                collector_port=self.collector_port,
                processing_capacity=capacity
            )
            workers.append(Worker(config))
        
        return workers