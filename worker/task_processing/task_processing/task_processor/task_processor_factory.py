from packages.enums import WorkLoad, WorkType, LoadingMode
from worker.task_processing.task_processing.task_processor.default_task_processor import DefaultTaskProcessor
from worker.task_processing.task_processing.task_processor.task_processor import TaskProcessor
from worker.task_processing.task_processing.task_processor.yolo.detection_image_processor import DetectionYOLOImageProcessor
from worker.task_processing.task_processing.task_processor.yolo.obb_image_processor import OBBYOLOImageProcessor
from worker.global_variables import GlobalVariables


class TaskProcessorFactory:
    @staticmethod
    def create_task_processor(work_type: WorkType,
                              work_load: WorkLoad,
                              model_loading_mode: LoadingMode) -> TaskProcessor:
        match work_type:
            case WorkType.YOLO_OBB:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11n-obb.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11s-obb.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'obb' / 'yolo11m-obb.pt',
                }

                return OBBYOLOImageProcessor(work_load, model_loading_mode, model_paths)
            case WorkType.YOLO_DETECTION:
                model_paths = {
                    'low': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11n.pt',
                    'medium': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11s.pt',
                    'high': GlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models' / 'detection' / 'yolo11m.pt',
                }

                return DetectionYOLOImageProcessor(work_load, model_loading_mode, model_paths)
            case WorkType.NONE:
                return DefaultTaskProcessor()