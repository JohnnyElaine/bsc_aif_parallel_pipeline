from aif_edge_worker.enums.computation_workload import ComputationWorkload
from aif_edge_worker.enums.computation_type import ComputationType
from aif_edge_worker.enums.loading_mode import LoadingMode
from aif_edge_worker.computation.image_processing.image_processor.default_image_processor import DefaultImageProcessor
from aif_edge_worker.computation.image_processing.image_processor.image_processor import ImageProcessor
from aif_edge_worker.computation.image_processing.image_processor.yolo.detection_image_processor import DetectionYOLOImageProcessor
from aif_edge_worker.computation.image_processing.image_processor.yolo.obb_image_processor import OBBYOLOImageProcessor
from aif_edge_worker.global_variables import GlobalVariables


class ImageProcessorFactory:
    @staticmethod
    def create_image_processor(computation_type: ComputationType,
                               computation_workload: ComputationWorkload,
                               model_loading_mode: LoadingMode) -> ImageProcessor:
        match computation_type:
            case ComputationType.YOLO_OBB:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11n-obb.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11s-obb.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11m-obb.pt',
                }

                return OBBYOLOImageProcessor(computation_workload, model_loading_mode, model_paths)
            case ComputationType.YOLO_DETECTION:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11n.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11s.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11m.pt',
                }

                return DetectionYOLOImageProcessor(computation_workload, model_loading_mode, model_paths)
            case ComputationType.NONE:
                return DefaultImageProcessor()