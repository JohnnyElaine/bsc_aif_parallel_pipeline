from aif_edge_node.image_processing.image_processor.detect_image_processor import DetectionImageProcessor
from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor
from aif_edge_node.image_processing.image_processor.obb_image_processor import OBBImageProcessor


class ImageProcessorFactory:
    @staticmethod
    def create_image_processor(type: str) -> ImageProcessor:
        match type:
            case "obb":
                return OBBImageProcessor()
            case "detection":
                return DetectionImageProcessor()
            case _:
                return DefaultImageProcessor()