import pickle
import pandas as pd
from threading import Event
from multiprocessing import Process
from queue import Queue


import packages.logging as logging
from producer.request_handling.request_handler import RequestHandler
from producer.data.task_config import TaskConfig
from packages.data import Video
from producer.elasticity.agent_pipeline import AgentPipeline
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.producer_config import ProducerConfig
from producer.task_generation.task_generator import TaskGenerator


class Producer(Process):
    def __init__(self, config: ProducerConfig, shared_stats_dict=None):
        """
        Initialize the coordinator with the video path and edge node information.
        :param port: port the producer listens on.
        :param video_path: Path to the input video file.
        """
        super().__init__()
        self.config = config

        self._stats = dict()
        if shared_stats_dict is not None:
            self._stats = shared_stats_dict

    def run(self):
        log = logging.setup_logging('producer')

        log.info('starting producer')

        src_video = Video(self.config.video_path)
        task_config = TaskConfig(self.config.work_type, self.config.max_work_load, src_video.resolution, src_video.fps)

        start_task_generation_event = Event()
        task_queue = Queue(maxsize=TaskGenerator.MAX_QUEUE_SIZE)

        request_handler = RequestHandler(self.config.port,
                                         task_queue,task_config.work_type, task_config.max_work_load,
                                         self.config.loading_mode, start_task_generation_event)
        task_generator = TaskGenerator(task_queue, src_video, start_task_generation_event)

        elasticity_handler = ElasticityHandler(task_config, task_generator, request_handler)
        slo_agent_pipeline = AgentPipeline(elasticity_handler, self.config.agent_type, request_handler.start_task_generation_event)

        request_handler.start()
        task_generator.start()
        slo_agent_pipeline.start()

        task_generator.join()
        request_handler.join()

        # SLO Agent serves no purpose when other threads are dead
        slo_agent_pipeline.stop()
        slo_agent_pipeline.join()

        # Collect statistics before shutting down
        self._stats['slo_stats'] = slo_agent_pipeline.get_slo_statistics()
        self._stats['worker_stats'] = request_handler.get_worker_statistics()

        log.info('stopped producer')

    def get_statistics(self) -> dict[str, pd.DataFrame]:
        """Retrieve and deserialize statistics"""
        return {
            'slo_stats': pickle.loads(self._stats['slo_stats']) if 'slo_stats' in self._stats else None,
            'worker_stats': pickle.loads(self._stats['worker_stats']) if 'worker_stats' in self._stats else None
        }