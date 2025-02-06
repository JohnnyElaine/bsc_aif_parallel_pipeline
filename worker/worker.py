from multiprocessing import Process, Pipe
from queue import Queue

import packages.logging as logging
from packages.enums import WorkType
from packages.enums import WorkLoad
from worker.communication.channel.request_channel import RequestChannel
from worker.communication.work_requester.network.zmq_work_requester import ZmqWorkRequester
from worker.enums.loading_mode import LoadingMode
from worker.global_variables import GlobalVariables
from worker.computation.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from worker.computation.task_processing.image.image_task_processor import ImageTaskProcessor
from worker.computation.task_processing.simulator.basic_stream_simulator import BasicStreamSimulator
from worker.communication.pipe_sender.pipe_sender import PipeSender
from worker.worker_config import WorkerConfig


class Worker(Process):
    def __init__(self, config: WorkerConfig):
        super().__init__()
        self.config = config

    def run(self):
        log = logging.setup_logging('worker')
        
        if self.config.is_simulation:
            self._run_simulation()
            return

        log.info(f"starting worker-{self.config.identity}")

        request_channel = RequestChannel(self.config.producer_ip, self.config.producer_port, self.config.identity)
        request_channel.connect()
        log.debug(f"established connection to producer-{request_channel}")

        work_config = self._register_at_producer(request_channel)

        if work_config is None:
            log.info(f'failed to register at producer {request_channel}')
            return

        log.info(f'registered at producer {request_channel}. Received config {work_config}')

        # Pipe for IPC
        pipe_receiving_end, pipe_sending_end = Pipe(False)
        log.debug('created pipe for IPC (Main Process -> Task Processor)')

        # create new process for image computation
        task_processor = self._create_task_processor(pipe_receiving_end,
                                                     work_config.work_type,
                                                     work_config.work_load)

        # create shared (frame buffer) queue for work_requester & pipe sender
        shared_queue = Queue()

        work_requester = ZmqWorkRequester(shared_queue, request_channel)
        pipe_sender = PipeSender(shared_queue, pipe_sending_end)

        task_processor.start()
        work_requester.start()
        pipe_sender.start()

        work_requester.join()
        pipe_sender.join()
        task_processor.join()

    def _register_at_producer(self, control_channel: RequestChannel):
        return control_channel.register()

    def _create_task_processor(self, pipe_receiving_end, work_type, work_load):
        image_processor = ImageProcessorFactory.create_image_processor(work_type, work_load, self.config.model_loading_mode)
        image_processor.initialize()
        return ImageTaskProcessor(self.config.identity, image_processor, pipe_receiving_end)

    def _run_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'
        image_processor = ImageProcessorFactory.create_image_processor(WorkType.YOLO_DETECTION, WorkLoad.MEDIUM, LoadingMode.LAZY)
        return BasicStreamSimulator(image_processor, input_video, True)