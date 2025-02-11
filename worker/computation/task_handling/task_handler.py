import cv2 as cv
import numpy as np
from multiprocessing import Process, Pipe, Event

import packages.logging as logging
from packages.data import TaskType, InstructionType
from packages.enums import WorkLoad
from worker.computation.task_processing.task_processor.task_processor_factory import TaskProcessorFactory
from worker.data.work_config import WorkConfig


class TaskHandler(Process):
    def __init__(self, identifier: int,
                 task_pipe_receiving_end: Pipe,
                 work_config: WorkConfig,
                 task_processor_initialized: Event):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_receiving_end
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

    def _iteration(self):
        """
        :return: True if the iteration was successful. False otherwise.
        """
        # receives dataclasses: Task, Instruction (must contain 'type' field)
        work = self._task_pipe.recv()

        match work.type:
            case TaskType.INFERENCE:
                frame = self._process_task(work.data)

                # Display the frame (optional)
                cv.imshow(f'Worker-{self.identifier}', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return False

            case InstructionType.CHANGE_WORK_LOAD:
                WorkLoad.int_to_enum(work.value)
                self._task_processor.change_work_load(work.value)
                self.log.debug(f'successfully changed work-load to {work.value}')
            case _:
                self.log.debug(f'task-processor received unknown work-type: {work.type}')
                pass

        return True

    def _process_task(self, task: np.ndarray):
        return self._task_processor.process(task)