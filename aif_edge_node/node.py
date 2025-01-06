from multiprocessing import Process

import logging

from aif_edge_node.enum.computation_type import ComputationType
from aif_edge_node.enum.stream_type import StreamType
from aif_edge_node.global_variables import GlobalVariables
from aif_edge_node.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from aif_edge_node.video_stream.local.local_message_stream_receiver import LocalMessageStreamReceiver
from aif_edge_node.video_stream.simulator.basic_stream_simulator import BasicStreamSimulator

log = logging.getLogger("aif_edge_node")


class Node:
    def __init__(self, identifier: int,
                 computation_type: ComputationType,
                 stream_type: StreamType,
                 port: int):
        self.identifier = identifier
        self._stream_type = stream_type
        self._image_processor = ImageProcessorFactory.create_image_processor(computation_type)
        self._port = port

    def start(self):
        log.debug(f"starting node-{self.identifier}")
        process = Process(target=self._run)
        process.start()
        return process

    def _run(self):
        stream = None

        if self._stream_type == StreamType.SIMULATION:
            stream = self._create_simulation()
        else:
            stream = self._create_stream_receiver()

        stream.start()


    def _create_stream_receiver(self):
        stream_receiver = LocalMessageStreamReceiver(self._port)
        return stream_receiver

    def _create_simulation(self):
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'

        return BasicStreamSimulator(self._image_processor, input_video, True)