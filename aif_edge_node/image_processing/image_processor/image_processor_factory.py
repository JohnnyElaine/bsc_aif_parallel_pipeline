from aif_edge_node.enum.computation_type import ComputationType
from aif_edge_node.image_processing.image_processor.default_image_processor import DefaultImageProcessor
from aif_edge_node.image_processing.image_processor.image_processor import ImageProcessor
from aif_edge_node.image_processing.image_processor.yolo.bb_image_processor import BBYOLOImageProcessor
from aif_edge_node.image_processing.image_processor.yolo.obb_image_processor import OBBYOLOImageProcessor


class ImageProcessorFactory:
    @staticmethod
    def create_image_processor(computation_type: ComputationType) -> ImageProcessor:
        match computation_type:
            case ComputationType.OBB:
                return OBBYOLOImageProcessor()
            case ComputationType.DETECTION:
                return BBYOLOImageProcessor()
            case ComputationType.NONE:
                return DefaultImageProcessor()