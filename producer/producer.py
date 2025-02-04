from multiprocessing import Process
from queue import Queue

import packages.logging as logging
from producer.communication.request_handler import RequestHandler
from producer.data.work_config import WorkConfig
from producer.producer_config import ProducerConfig
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

        # create shared (frame buffer) queue for task generator & request handler
        shared_queue = Queue()
        request_handler = RequestHandler(self.config.port,
                                         shared_queue,
                                         WorkConfig(self.config.worker_type, self.config.work_load))
        task_generator = TaskGenerator(shared_queue, self.config.video_path)

        request_handler.start()
        task_generator.start()


        task_generator.join()
        request_handler.join()

