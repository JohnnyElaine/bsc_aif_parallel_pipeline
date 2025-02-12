import cv2 as cv
from numpy import ndarray
from multiprocessing import Process, Pipe, Event

import packages.logging as logging
from packages.data import TaskType, InstructionType, Task
from packages.enums import WorkLoad
from worker.task_processing.task_processing.task_processor.task_processor_factory import TaskProcessorFactory
from worker.data.work_config import WorkConfig


class TaskProcessingPipeline(Process):
    def __init__(self, identifier: int,
                 work_config: WorkConfig,
                 task_processor_initialized: Event, task_pipe_recv_end: Pipe, result_pipe_send_end: Pipe):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_recv_end
        self._result_pipe = result_pipe_send_end
        self._work_config = work_config
        self._task_processor_initialized = task_processor_initialized
        self._task_processor = None
        self._is_running = False
        self.log = None

    def run(self):
        self.log = logging.setup_logging('task_handler')
        self.log.debug('initializing image-task-processor')
        self._task_processor = TaskProcessorFactory.create_task_processor(self._work_config.work_type,
                                                                          self._work_config.work_load,
                                                                          self._work_config.loading_mode)
        self._task_processor.initialize()
        self._task_processor_initialized.set()

        self.log.debug('starting task-handler')
        self._is_running = True
        try:
            while self._is_running:
                ok = self._iteration()
                if not ok:
                    break
        except EOFError:
            print('Producer disconnected. Node exiting.')

        self.stop()

    def stop(self):
        self.log.info('stopping task-handler')
        self._is_running = False

    def _iteration(self) -> bool:
        """
        :return: True if the iteration was successful. False otherwise.
        """
        # receives dataclasses: Task, Instruction (must contain 'type' field)
        work = self._task_pipe.recv()

        match work.type:
            case TaskType.INFERENCE:
                processed_data = self._process_task(work.data)

                # Display the frame (optional)
                #self._display_frame(processed_data)

                result = Task(work.id, TaskType.COLLECT, processed_data)

                self._result_pipe.send(result)

            case InstructionType.CHANGE_WORK_LOAD:
                WorkLoad.int_to_enum(work.value)
                self._task_processor.change_work_load(work.value)
                self.log.debug(f'successfully changed work-load to {work.value}')
            case _:
                self.log.debug(f'task-processor received unknown work-type: {work.type}')
                pass

        return True

    def _process_task(self, task: ndarray):
        return self._task_processor.process(task)

    def _display_frame(self, frame: ndarray):
        cv.imshow(f'Worker-{self.identifier}', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False