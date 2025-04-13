import time
from multiprocessing import Process, Pipe, Event

import cv2 as cv
from numpy import ndarray

import packages.logging as logging
from packages.data import TaskType, Task
from packages.enums import WorkLoad
from worker.data.work_config import WorkConfig
from worker.task_processing.task_processing.task_processor.task_processor_factory import TaskProcessorFactory


class TaskProcessingPipeline(Process):
    def __init__(self, identifier: int,
                 work_config: WorkConfig,
                 task_processor_initialized: Event,
                 task_pipe_recv_end: Pipe,
                 result_pipe_send_end: Pipe,
                 artificial_processing_capacity: float):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_recv_end
        self._result_pipe = result_pipe_send_end
        self._work_config = work_config
        self._task_processor_initialized = task_processor_initialized
        self._processing_capacity = artificial_processing_capacity # artificially limit capacity (1 = 100%, 0.5 = 50%, etc)
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
                processed_data = self._process_task_function(task.data)

                # Display the frame (optional)
                # self._display_frame(processed_data)

                result = Task(TaskType.COLLECT, task.id, processed_data)

                self._result_pipe.send(result)

            case TaskType.CHANGE_WORK_LOAD:
                w = WorkLoad.int_to_enum(task.data.item())
                self._task_processor.change_work_load(w)
                self.log.info(f'successfully changed work-load to {w}')

            case TaskType.END:
                self._result_pipe.send(task) # notify collector that the transmission has ended
                return False # stop the main process loop

            case _:
                self.log.debug(f'task-processing-pipeline received unknown work-type: {task.type}')
                pass

        return True

    def _process_task(self, task: ndarray) -> ndarray:
        return self._task_processor.process(task)

    def _process_task_with_delay(self, task: ndarray) -> ndarray:
        start_time = time.perf_counter()
        processed_task = self._task_processor.process(task)
        processing_time = time.perf_counter() - start_time
        time.sleep(processing_time / self._processing_capacity - processing_time) # sleep to simulate less processing power

        return processed_task

    def _display_frame(self, frame: ndarray):
        cv.imshow(f'Worker-{self.identifier}', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False