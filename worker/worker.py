from multiprocessing import Process, Pipe
from queue import Queue

import packages.logging as logging
from packages.enums import ComputeType
from packages.enums import ComputeLoad
from worker.communication.channel.request_channel import RequestChannel
from worker.communication.work_requester.network.zmq_work_requester import ZmqWorkRequester
from worker.enums.loading_mode import LoadingMode
from worker.enums.work_source import WorkSource
from worker.global_variables import GlobalVariables
from worker.computation.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from worker.computation.task_processing.image.image_task_processor import ImageTaskProcessor
from worker.computation.task_processing.simulator.basic_stream_simulator import BasicStreamSimulator
from worker.communication.old_work_receiver.local.local_pull_work_receiver import LocalWorkReceiver
from worker.communication.pipe_sender.pipe_sender import PipeSender
from worker.worker_config import WorkerConfig


class Worker(Process):
    def __init__(self, config: WorkerConfig):
        super().__init__()
        self._config = config
        self._stream_config = None

    def run(self):
        log = logging.setup_logging('worker')
        
        if self._config.is_simulation:
            self._run_simulation()
            return

        log.info(f"starting worker-{self._config.identity}")

        request_channel = RequestChannel(self._config.producer_ip, self._config.producer_port, self._config.identity)
        request_channel.connect()
        log.debug(f"established connection to producer-{request_channel}")

        if self._register_at_producer(request_channel) is None:
            log.info(f'failed to register at producer {request_channel}')
            return

        # Pipe for IPC
        log.debug('creating pipe for IPC (Main Process -> Stream-Computer)')
        pipe_receiving_end, pipe_sending_end = Pipe(False)

        # create new process for image computation
        task_processor = self._create_task_processor(pipe_receiving_end,
                                                     self._stream_config.compute_type,
                                                     self._stream_config.compute_load)

        # create shared (frame buffer) queue for work_requester & pipe sender
        shared_queue = Queue()

        work_requester = ZmqWorkRequester(shared_queue, request_channel)
        pipe_sender = PipeSender(shared_queue, pipe_sending_end)

        work_requester.start()
        pipe_sender.start()
        task_processor.start()

        # avoid killing the main thread
        work_requester.join()
        pipe_sender.join()
        task_processor.join()

    def _register_at_producer(self, control_channel: RequestChannel):
        return control_channel.register()

    def _create_task_processor(self, pipe_receiving_end, compute_type, compute_load):
        image_processor = ImageProcessorFactory.create_image_processor(compute_type, compute_load, self._config.model_loading_mode)
        image_processor.initialize()
        return ImageTaskProcessor(self._config.identity, image_processor, pipe_receiving_end)

    def _create_stream_receiver(self, shared_queue: Queue):
        match self._stream_config.stream_source:
            case WorkSource.LOCAL_MESSAGE:
                return LocalWorkReceiver(self._config.producer_port, shared_queue)
            case WorkSource.ZEROMQ_RADIO_DISH:
                return
            case WorkSource.ZEROMQ_LOAD_BALANCING:
                return

    def _run_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'
        image_processor = ImageProcessorFactory.create_image_processor(ComputeType.YOLO_DETECTION, ComputeLoad.MEDIUM, LoadingMode.LAZY)
        return BasicStreamSimulator(image_processor, input_video, True)