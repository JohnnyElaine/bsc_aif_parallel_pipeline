from packages.enums import WorkType
from worker.data.work_config import WorkConfig
from worker.global_variables import WorkerGlobalVariables
from worker.task_processing.task_processing.task_processor.default_task_processor import DefaultTaskProcessor
from worker.task_processing.task_processing.task_processor.yolo.detection_image_processor import \
    DetectionYOLOImageProcessor
from worker.task_processing.task_processing.task_processor.yolo.obb_image_processor import OBBYOLOImageProcessor


class TaskProcessorFactory:
    @staticmethod
    def create_task_processor(work_config: WorkConfig):
        
        models_path = WorkerGlobalVariables.PROJECT_ROOT / 'checkpoints' / 'models'

        match work_config.work_type:
            case WorkType.YOLO_OBB:
                model_paths = {
                    'low': models_path / 'obb' / 'yolo11n-obb.pt',
                    'medium': models_path / 'obb' / 'yolo11s-obb.pt',
                    'high': models_path / 'obb' / 'yolo11m-obb.pt',
                }

                return OBBYOLOImageProcessor(work_config.inference_quality, work_config.loading_mode, model_paths)

            case WorkType.YOLO_DETECTION:
                model_paths = {
                    'low': models_path / 'detection' / 'yolo11n.pt',
                    'medium': models_path / 'detection' / 'yolo11s.pt',
                    'high': models_path / 'detection' / 'yolo11m.pt',
                }

                return DetectionYOLOImageProcessor(work_config.inference_quality, work_config.loading_mode, model_paths)
            
            case WorkType.NONE:
                return DefaultTaskProcessor()
        return None