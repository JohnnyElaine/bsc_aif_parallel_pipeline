from multiprocessing import Process
from queue import Queue

import packages.logging as logging
from producer.communication.request_handler import RequestHandler
from producer.data.task_config import TaskConfig
from producer.data.video import Video
from producer.elasticity.handler.elasticity_handler import ElasticityHandler
from producer.producer_config import ProducerConfig
from producer.elasticity.agent_pipeline import AgentPipeline
from producer.task_generation.task_generator import TaskGenerator


class Producer(Process):
    def __init__(self, config: ProducerConfig):
        """
        Initialize the coordinator with the video path and edge node information.
        :param port: port the producer listens on.
        :param video_path: Path to the input video file.
        """
        super().__init__()
        self.config = config

    def run(self):
        log = logging.setup_logging('producer')

        log.info("starting producer")

        src_video = Video(self.config.video_path)
        task_config = TaskConfig(self.config.work_type, self.config.max_work_load, src_video.resolution, src_video.fps)

        task_queue = Queue(maxsize=TaskGenerator.MAX_QUEUE_SIZE)
        request_handler = RequestHandler(self.config.port,
                                         task_queue,
                                         task_config.work_type, task_config.max_work_load, self.config.loading_mode)
        task_generator = TaskGenerator(task_queue, src_video, request_handler.start_task_generator_event)

        elasticity_handler = ElasticityHandler(task_config, task_generator, request_handler)
        slo_agent = AgentPipeline(elasticity_handler, self.config.agent_type, request_handler.start_task_generator_event)

        request_handler.start()
        task_generator.start()
        slo_agent.start()

        task_generator.join()
        request_handler.join()