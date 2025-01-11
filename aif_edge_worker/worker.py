from multiprocessing import Process, Pipe
from queue import Queue

from aif_edge_worker.communication.control.control_channel.control_channel import ControlChannel
from aif_edge_worker.communication.control.control_channel.req import req_builder
from aif_edge_worker.communication.control.control_channel.stream_config import StreamConfig
from aif_edge_worker.enums.computation_type import ComputationType
from aif_edge_worker.enums.computation_workload import ComputationWorkload
from aif_edge_worker.enums.loading_mode import LoadingMode
from aif_edge_worker.enums.stream_source import StreamSource
from aif_edge_worker.global_variables import GlobalVariables
from aif_edge_worker.computation.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from aif_edge_worker.computation.stream_computation.general.local_message_stream_computer import GeneralStreamComputer
from aif_edge_worker.computation.stream_computation.simulator.basic_stream_simulator import BasicStreamSimulator
from aif_edge_worker.communication.data.stream_receiver.local.local_stream_receiver import LocalStreamReceiver
from aif_edge_worker.communication.data.stream_receiver.pipe_sender.pipe_sender import PipeSender
from shared.setup_logging import setup_logging


class Worker(Process):
    def __init__(self, identifier: int,
                 controller_ip: str,
                 controller_port: int,
                 is_simulation=False):
        super().__init__()
        self._identifier = identifier
        self._controller_ip = controller_ip
        self._controller_port = controller_port
        self._stream_config = None
        self._log = setup_logging('worker')
        self._is_simulation = is_simulation

    def run(self):
        if self._is_simulation:
            self._run_simulation()
            return

        control_channel = ControlChannel(self._controller_ip, self._controller_port)
        control_channel.connect()

        if not self._register_at_controller(control_channel):
            return
        
        self._log.info(f"starting node-{self._identifier}")

        # Pipe for IPC
        self._log.debug('creating pipe for IPC (Main Process -> Stream-Computer)')
        pipe_receiving_end, pipe_sending_end = Pipe(False)

        # create processes (computation needs more resources)
        stream_computer = self._create_stream_computer(pipe_receiving_end, self)
        stream_computer.start()

        # create threads
        shared_queue = Queue()

        pipe_sender = PipeSender(shared_queue, pipe_sending_end)
        pipe_sender.start()

        # run instead of start to avoid killing this thread
        stream_receiver = LocalStreamReceiver(self._controller_port, shared_queue)
        stream_receiver.run()

    def _register_at_controller(self, control_channel: ControlChannel):
        req_register = req_builder.register(self._identifier)
        rep = control_channel.send_req(req_register)

        if not rep['success']:
            self._log.error(f"worker-{self._identifier} failed to register at controller {self._identifier}:{self._controller_port}")
            return False

        self._stream_config = StreamConfig(
            rep['video_fps'],
            rep['video_resolution'],
            rep['computation_type'],
            rep['computation_workload'],
            rep['stream_source']
        )

        return True

    def _create_stream_computer(self, pipe_receiving_end, computation_type, computation_workload):
        image_processor = ImageProcessorFactory.create_image_processor(computation_type, computation_workload,)
        image_processor.initialize()
        return GeneralStreamComputer(self._identifier, image_processor, pipe_receiving_end)

    def _create_stream_receiver(self, shared_queue: Queue):
        match self._stream_config.stream_source:
            case StreamSource.LOCAL_MESSAGE:
                return LocalStreamReceiver(self._controller_port, shared_queue)
            case StreamSource.ZEROMQ_RADIO_DISH:
                return
            case StreamSource.ZEROMQ_LOAD_BALANCING:
                return

    def _run_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'
        image_processor = ImageProcessorFactory.create_image_processor(ComputationType.YOLO_DETECTION, ComputationWorkload.MEDIUM, LoadingMode.LAZY)
        return BasicStreamSimulator(image_processor, input_video, True)