from multiprocessing import Process, Pipe
from queue import Queue

from worker.communication.channel.request_channel import RequestChannel
from worker.communication.old_push_control.control_channel.pull_control_channel import PullControlChannel
from worker.communication.old_push_control.control_channel.pull_stream_config import PullStreamConfig
from worker.communication.work_requester.work_requester import WorkRequester
from worker.data.stream_config import StreamConfig
from worker.enums.compute_type import ComputeType
from worker.enums.compute_load import ComputeLoad
from worker.enums.loading_mode import LoadingMode
from worker.enums.stream_source import StreamSource
from worker.global_variables import GlobalVariables
from worker.computation.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from worker.computation.stream_computation.general.local_message_stream_computer import GeneralStreamComputer
from worker.computation.stream_computation.simulator.basic_stream_simulator import BasicStreamSimulator
from worker.communication.old_work_receiver.local.local_pull_work_receiver import LocalWorkReceiver
from worker.communication.pipe_sender.pipe_sender import PipeSender
from shared.setup_logging import setup_logging


class Worker(Process):
    def __init__(self,
                 identity: int,
                 model_loading_mode: LoadingMode,
                 producer_ip: str,
                 producer_port: int,
                 is_simulation=False):
        super().__init__()
        self._id = identity
        self._model_loading_mode = model_loading_mode
        self._producer_ip = producer_ip
        self._producer_port = producer_port
        self._stream_config = None
        self._log = setup_logging('worker')
        self._is_simulation = is_simulation

    def run(self):
        if self._is_simulation:
            self._run_simulation()
            return

        self._log.info(f"starting worker-{self._id}")

        request_channel = RequestChannel(self._producer_ip, self._producer_port, self._id)
        self._log.debug(f"connecting to-{request_channel}")
        request_channel.connect()

        if not self._register_at_producer(request_channel):
            return

        # Pipe for IPC
        self._log.debug('creating pipe for IPC (Main Process -> Stream-Computer)')
        pipe_receiving_end, pipe_sending_end = Pipe(False)

        # create processes (computation needs more resources)
        stream_computer = self._create_stream_computer(pipe_receiving_end,
                                                       self._stream_config.compute_type,
                                                       self._stream_config.compute_load)
        stream_computer.start()

        # create threads
        shared_queue = Queue()

        pipe_sender = PipeSender(shared_queue, pipe_sending_end)
        pipe_sender.start()

        # run instead of start to avoid killing this thread
        work_requester = WorkRequester(shared_queue)
        work_requester.start()

        work_requester.join()
        pipe_sender.join()
        stream_computer.join()

    def _register_at_producer(self, control_channel: RequestChannel):
        val = control_channel.register()

        if val is None:
            return False

        self._stream_config = val

        return True

    def _create_stream_computer(self, pipe_receiving_end, compute_type, compute_load):
        image_processor = ImageProcessorFactory.create_image_processor(compute_type, compute_load, self._model_loading_mode)
        image_processor.initialize()
        return GeneralStreamComputer(self._id, image_processor, pipe_receiving_end)

    def _create_stream_receiver(self, shared_queue: Queue):
        match self._stream_config.stream_source:
            case StreamSource.LOCAL_MESSAGE:
                return LocalWorkReceiver(self._producer_port, shared_queue)
            case StreamSource.ZEROMQ_RADIO_DISH:
                return
            case StreamSource.ZEROMQ_LOAD_BALANCING:
                return

    def _run_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'
        image_processor = ImageProcessorFactory.create_image_processor(ComputeType.YOLO_DETECTION, ComputeLoad.MEDIUM, LoadingMode.LAZY)
        return BasicStreamSimulator(image_processor, input_video, True)