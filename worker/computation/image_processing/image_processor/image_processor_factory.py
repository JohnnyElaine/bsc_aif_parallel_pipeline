from worker.enums.compute_load import ComputeLoad
from worker.enums.compute_type import ComputeType
from worker.enums.loading_mode import LoadingMode
from worker.computation.image_processing.image_processor.default_image_processor import DefaultImageProcessor
from worker.computation.image_processing.image_processor.image_processor import ImageProcessor
from worker.computation.image_processing.image_processor.yolo.detection_image_processor import DetectionYOLOImageProcessor
from worker.computation.image_processing.image_processor.yolo.obb_image_processor import OBBYOLOImageProcessor
from worker.global_variables import GlobalVariables


class ImageProcessorFactory:
    @staticmethod
    def create_image_processor(computation_type: ComputeType,
                               compute_load: ComputeLoad,
                               model_loading_mode: LoadingMode) -> ImageProcessor:
        match computation_type:
            case ComputeType.YOLO_OBB:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11n-obb.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11s-obb.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11m-obb.pt',
                }

                return OBBYOLOImageProcessor(compute_load, model_loading_mode, model_paths)
            case ComputeType.YOLO_DETECTION:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11n.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11s.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11m.pt',
                }

                return DetectionYOLOImageProcessor(compute_load, model_loading_mode, model_paths)
            case ComputeType.NONE:
                return DefaultImageProcessor()