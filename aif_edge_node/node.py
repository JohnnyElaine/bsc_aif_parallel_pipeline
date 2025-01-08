import logging

from multiprocessing import Process, Pipe
from queue import Queue

from aif_edge_node.enums.computation_type import ComputationType
from aif_edge_node.enums.stream_type import StreamType
from aif_edge_node.global_variables import GlobalVariables
from aif_edge_node.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from aif_edge_node.stream_computation.general.local_message_stream_computer import GeneralStreamComputer
from aif_edge_node.stream_computation.simulator.basic_stream_simulator import BasicStreamSimulator
from aif_edge_node.stream_receiver.local.local_stream_receiver import LocalStreamReceiver
from aif_edge_node.stream_receiver.pipe_sender.pipe_sender import PipeSender
from shared.setup_logging import setup_logging


class Node(Process):
    def __init__(self, identifier: int,
                 computation_type: ComputationType,
                 stream_type: StreamType,
                 port: int):
        super().__init__()
        self.identifier = identifier
        self._stream_type = stream_type
        self._image_processor = ImageProcessorFactory.create_image_processor(computation_type)
        self._port = port

    def run(self):
        log = setup_logging('node')
        
        log.debug(f"starting node-{self.identifier}")
        self._image_processor.initialize()

        # Pipe for IPC
        log.debug('creating pipe for IPC (Main Process -> Stream-Computer)')
        pipe_receiving_end, pipe_sending_end = Pipe(False)

        # create processes (computation needs more resources)
        stream_computer = self._create_stream_computer(pipe_receiving_end)
        stream_computer.start()

        # create threads
        shared_queue = Queue()

        pipe_sender = PipeSender(shared_queue, pipe_sending_end)
        pipe_sender.start()

        # run to avoid killing this thread
        stream_receiver = LocalStreamReceiver(self._port, shared_queue)
        stream_receiver.run()

    def _create_stream_computer(self, pipe_receiving_end):
        if self._stream_type == StreamType.SIMULATION:
            return self._create_simulation()

        return GeneralStreamComputer(self.identifier, self._image_processor, pipe_receiving_end)

    def _create_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'

        return BasicStreamSimulator(self._image_processor, input_video, True)