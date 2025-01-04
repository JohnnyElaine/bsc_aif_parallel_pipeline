import logging
from aif_edge_node.global_variables import GlobalVariables

from aif_edge_node.image_processing.image_processor.image_processor_factory import ImageProcessorFactory
from aif_edge_node.video_stream.simulator.basic_stream_simulator import BasicStreamSimulator

log = logging.getLogger("aif_edge_node")


class Node:
    def __init__(self, name: str, computation_type: str, simulation: bool):
        self.name = name
        self.computation_type = computation_type
        self.simulation = simulation
        self.image_processor = ImageProcessorFactory.create_image_processor(computation_type)


    def start(self):
        log.debug("Starting node")
        # input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '4K Video of Highway Traffic! [KBsqQez-O4w].mp4'
        input_video = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'obb' / 'Video Background Stock Footage Free ( Port, yachts, flying by a drone on the piers and marinas ) [XISqY-EC-QQ].mp4'

        stream_simulator = BasicStreamSimulator(self.image_processor, input_video, True)
        stream_simulator.start()