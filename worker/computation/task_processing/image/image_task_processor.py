import logging


import cv2 as cv
import numpy as np


from multiprocessing import Process, Pipe

from worker.computation.image_processing.image_processor.image_processor import ImageProcessor
from worker.computation.task_processing.task_processor import TaskProcessor


log = logging.getLogger("worker")


class ImageTaskProcessor(Process, TaskProcessor):
    def __init__(self, identifier: int, image_processor: ImageProcessor, pipe_receiving_end: Pipe):
        super().__init__()
        self.identifier = identifier
        self._pipe_receiving_end = pipe_receiving_end
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
        task = self._pipe_receiving_end.recv()

        frame = self._process_task(task.task)

        # Display the frame (optional)
        cv.imshow(f'Worker-{self.identifier}', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def _take_message_from_receiver(self):
        data = self._pipe_receiving_end.recv()
        return data['frame'], data['frame_index']

    def _process_task(self, task: np.ndarray):
        return self._image_processor.process_image(task)

