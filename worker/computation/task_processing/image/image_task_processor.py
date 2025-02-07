import logging
import cv2 as cv
import numpy as np
from multiprocessing import Process, Pipe

from packages.data import TaskType, InstructionType
from worker.computation.image_processing.image_processor.image_processor import ImageProcessor
from worker.computation.task_processing.task_processor import TaskProcessor

log = logging.getLogger("worker")


class ImageTaskProcessor(Process, TaskProcessor):
    def __init__(self, identifier: int, image_processor: ImageProcessor, task_pipe_receiving_end: Pipe):
        super().__init__()
        self.identifier = identifier
        self._task_pipe = task_pipe_receiving_end
        self._image_processor = image_processor
        self._is_running = False

    def run(self):
        log.debug("starting image-task-processor")
        self._is_running = True
        while self._is_running:
            try:
                ok = self._iteration()
                if not ok:
                    break
            except EOFError:
                print("Producer disconnected. Node exiting.")
                break

        self.stop()

    def stop(self):
        log.info("stopping image-stream-computer")
        self._is_running = False

    def _iteration(self):
        """
        :return: True if the iteration was successful. False otherwise.
        """
        work = self._task_pipe.recv()

        match work.type:
            case TaskType.INFERENCE:
                frame = self._process_task(work.data)

                # Display the frame (optional)
                cv.imshow(f'Worker-{self.identifier}', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return False
            case InstructionType.CHANGE_WORK_LOAD:
                self._image_processor.change_work_load(work.value)




        return True

    def _process_task(self, task: np.ndarray):
        return self._image_processor.process_image(task)