from aif_edge_node.image_processing.image_processor.default_image_processor import DefaultImageProcessor
from aif_edge_node.image_processing.image_processor.yolo.bb_image_processor import BBYOLOImageProcessor
from aif_edge_node.image_processing.image_processor.yolo.yolo_image_processor import YOLOImageProcessor
from aif_edge_node.image_processing.image_processor.yolo.obb_image_processor import OBBYOLOImageProcessor

class ImageProcessorFactory:
    @staticmethod
    def create_image_processor(type: str) -> YOLOImageProcessor:
        match type:
            case "obb":
                return OBBYOLOImageProcessor()
            case "detection":
                return BBYOLOImageProcessor()
            case _:
                return DefaultImageProcessor()