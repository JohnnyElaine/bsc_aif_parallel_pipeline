from aif_edge_worker.global_variables import GlobalVariables
from aif_edge_worker.enums.computation_type import ComputationType
from aif_edge_worker.worker import Worker
from aif_edge_worker.enums.stream_source import StreamSource
from controller.controller import Controller
from controller.communication.data.stream_generation.node_info.node_info import NodeInfo


def create_nodes(num: int, port: int):
    nodes = []
    nodes_info = []
    for i in range(num):
        nodes.append(Worker(i, ComputationType.YOLO_DETECTION, StreamSource.LOCAL_MESSAGE, port))
        nodes_info.append(NodeInfo(i, 'localhost', 0))

    return nodes, nodes_info

def main():
    port = 10000
    vid_path = GlobalVariables.PROJECT_ROOT / 'media' / 'vid' / 'general_detection' / '1080p Video of Highway Traffic! [KBsqQez-O4w].mp4'
    num_nodes = 1

    nodes, nodes_info = create_nodes(num_nodes, port)

    for node in nodes:
        node.start()

    controller = Controller(port, vid_path, nodes_info)
    controller.run()



if __name__ == "__main__":
    main()
