import logging
import time
from multiprocessing import Process, Pipe, Event, Value

from numpy import ndarray

import packages.logging as logging
from packages.data import TaskType, Task
from packages.enums import InferenceQuality
from worker.data.work_config import WorkConfig
from worker.task_processing.task_processing.task_processor.task_processor_factory import TaskProcessorFactory


class TaskProcessingPipeline(Process):
    def __init__(self, identifier: int,
                 work_config: WorkConfig,
                 task_processor_initialized: Event,
                 task_pipe_recv_end: Pipe,
                 result_pipe_send_end: Pipe,
                 artificial_processing_capacity: float,
                 latest_processing_time: Value):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_recv_end
        self._result_pipe = result_pipe_send_end
        self._work_config = work_config
        self._task_processor_initialized = task_processor_initialized
        self._processing_capacity = artificial_processing_capacity # artificially limit capacity (1 = 100%, 0.5 = 50%, etc)
        self._latest_processing_time = latest_processing_time  # Shared value for processing time
        self._task_processor = None
        self._is_running = False
        self.log = None

        # artificially limit capacity by adding a delay for simulation testing
        self._process_task_function = self._process_task_with_delay if artificial_processing_capacity < 1 else self._process_task

    def run(self):
        self.log = logging.setup_logging('task_processing')
        self.log.debug('initializing task-processing-pipeline')
        self._task_processor = TaskProcessorFactory.create_task_processor(self._work_config)
        self._task_processor.initialize()
        self._task_processor_initialized.set()

        self.log.debug('starting task-processing-pipeline')
        self._is_running = True
        try:
            ok = True
            while self._is_running and ok:
                ok = self._iteration()
        except EOFError:
            print('Producer disconnected. Node exiting.')

        self.log.info('stopped task-processing-pipeline')

    def stop(self):
        self.log.info('stopping task-processing-pipeline')
        self._is_running = False

    def _iteration(self) -> bool:
        """
        :return: True if the iteration was successful. False otherwise.
        """
        # receives dataclasses: LocalMessage (Task, Change, Signal). Object must contain 'type' field
        task = self._task_pipe.recv()

        match task.type:
            case TaskType.INFERENCE:
                self._inference(task)

            case TaskType.CHANGE_INFERENCE_QUALITY:
                self._change_inference_quality(task)

            case TaskType.END:
                self._result_pipe.send(task) # notify collector that the transmission has ended
                return False # stop the main process loop

            case _:
                self.log.debug(f'task-processing-pipeline received unknown work-type: {task.type}')
                pass

        return True

    def _inference(self, task: Task):
        processing_start_t = time.perf_counter()
        processed_data = self._process_task_function(task.data)
        result = Task(TaskType.COLLECT, task.id, task.stream_key, processed_data)
        processing_t = time.perf_counter() - processing_start_t
        
        # Update shared processing time
        with self._latest_processing_time.get_lock():
            self._latest_processing_time.value = processing_t
        
        self._result_pipe.send(result)

    def _change_inference_quality(self, task):
        w = InferenceQuality.int_to_enum(task.data.item())
        self._task_processor.change_inference_quality(w)
        self.log.info(f'successfully changed inference-quality to {w}')

    def _process_task(self, task: ndarray) -> ndarray:
        return self._task_processor.process(task)

    def _process_task_with_delay(self, task: ndarray) -> ndarray:
        start_time = time.perf_counter()
        processed_task = self._process_task(task)
        processing_time = time.perf_counter() - start_time
        time.sleep(processing_time / self._processing_capacity - processing_time) # sleep to simulate less processing power


        return processed_task