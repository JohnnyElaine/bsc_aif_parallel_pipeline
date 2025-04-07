import time

import cv2 as cv
from numpy import ndarray
from multiprocessing import Process, Pipe, Event

import packages.logging as logging
from packages.data import TaskType, ChangeType, Task
from packages.data.types.signal_type import SignalType
from packages.enums import WorkLoad
from worker.task_processing.task_processing.task_processor.task_processor_factory import TaskProcessorFactory
from worker.data.work_config import WorkConfig


class TaskProcessingPipeline(Process):
    def __init__(self, identifier: int,
                 work_config: WorkConfig,
                 task_processor_initialized: Event,
                 task_pipe_recv_end: Pipe,
                 result_pipe_send_end: Pipe,
                 process_delay_s: float):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_recv_end
        self._result_pipe = result_pipe_send_end
        self._work_config = work_config
        self._task_processor_initialized = task_processor_initialized
        self._process_delay = process_delay_s
        self._task_processor = None
        self._is_running = False
        self.log = None

        # add artifical delay for simulation testing
        self._process_task_function = self._process_task if process_delay_s <= 0 else self._process_task_with_delay

    def run(self):
        self.log = logging.setup_logging('task_handler')
        self.log.debug('initializing task-processor')
        self._task_processor = TaskProcessorFactory.create_task_processor(self._work_config)
        self._task_processor.initialize()
        self._task_processor_initialized.set()

        self.log.debug('starting task-handler')
        self._is_running = True
        try:
            ok = True
            while self._is_running and ok:
                ok = self._iteration()
        except EOFError:
            print('Producer disconnected. Node exiting.')

        self.log.info('stopped task-handler')

    def stop(self):
        self.log.info('stopping task-handler')
        self._is_running = False

    def _iteration(self) -> bool:
        """
        :return: True if the iteration was successful. False otherwise.
        """
        # receives dataclasses: LocalMessage (Task, Change, Signal). Object must contain 'type' field
        msg = self._task_pipe.recv()

        match msg.type:
            case SignalType.END:
                self._result_pipe.send(msg) # notify collector that the transmission has ended
                return False # stop the main proces loop

            case TaskType.INFERENCE:
                processed_data = self._process_task_function(msg.data)

                # Display the frame (optional)
                # self._display_frame(processed_data)

                result = Task(TaskType.COLLECT, msg.id, processed_data)

                self._result_pipe.send(result)

            case ChangeType.CHANGE_WORK_LOAD:
                WorkLoad.int_to_enum(msg.value)
                self._task_processor.change_work_load(msg.value)
                self.log.debug(f'successfully changed msg-load to {msg.value}')
            case _:
                self.log.debug(f'task-processor received unknown msg-type: {msg.type}')
                pass

        return True

    def _process_task(self, task: ndarray):
        return self._task_processor.process(task)

    def _process_task_with_delay(self, task: ndarray):
        time.sleep(self._process_delay)
        return self._task_processor.process(task)

    def _display_frame(self, frame: ndarray):
        cv.imshow(f'Worker-{self.identifier}', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False