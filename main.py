import logging.config
import logging.handlers
import pathlib
import json
import atexit
from image_detector import YoloDetector
from video_stream import StreamSimulator

log = logging.getLogger("my_app")

def setup_logging(enable_file_logging=False):
    config_file = pathlib.Path("logging/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")

    if not enable_file_logging:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)

    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


def main():
    setup_logging(enable_file_logging=False)
    logging.basicConfig(level="DEBUG")

    yolo_detector = YoloDetector("./checkpoints/models/obb/yolo11s-obb.pt")

    stream_simulator = StreamSimulator('media/vid/Aerial view of a highway overpass [EORUDBFLBTM].mp4', yolo_detector)
    stream_simulator.start()


# Program Entry Point
if __name__ == "__main__":
    main()
