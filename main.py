import logging
from pathlib import Path

from aif_edge_node.global_variables import GlobalVariables
from aif_edge_node.enums.computation_type import ComputationType
from aif_edge_node.node import Node
from aif_edge_node.enums.stream_type import StreamType
from controller.controller import Controller
from controller.stream_generation.node_info.node_info import NodeInfo


def setup_logging(log_to_file=False, log_file_path=None):
    # Define a log format
    log_format = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt=date_format,
    )

    # Get a logger instance
    logger = logging.getLogger("aif_edge_node")

    # If file logging is enabled, add a file handler
    if log_to_file:
        if log_file_path is None:
            log_file_path = GlobalVariables.PROJECT_ROOT / 'log' / 'aif_edge_node.log'

        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        file_handler.setLevel(logging.INFO)  # Log INFO and above to the file
        logger.addHandler(file_handler)

    logger.debug("logger loaded successfully")

    return logger

def create_nodes(num: int, port: int):
    nodes = []
    nodes_info = []
    for i in range(num):
        nodes.append(Node(i, ComputationType.DETECTION, StreamType.LOCAL_MESSAGE, port))
        nodes_info.append(NodeInfo(i, 'localhost', 0))

    return nodes, nodes_info

def main():
    setup_logging()

    port = 5000
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    num_nodes = 1

    nodes, nodes_info = create_nodes(num_nodes, port)
    controller = Controller(port, vid_path, nodes_info)

    controller.start()
    for node in nodes:
        node.start()


if __name__ == "__main__":
    main()
