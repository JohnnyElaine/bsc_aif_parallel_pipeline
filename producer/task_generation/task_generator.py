import logging
import time
from threading import Thread
from queue import Queue

from packages.data import Task, TaskType
from producer.data.video import Video

log = logging.getLogger("producer")


class TaskGenerator(Thread):
    def __init__(self, shared_queue: Queue, video_path: str, registration_event):
        super().__init__()
        self._queue = shared_queue
        self._video_path = video_path
        self._video = None
        self._target_frame_time = None
        self.worker_ready_event = registration_event

    def run(self):
        self._video = Video(self._video_path)
        self._target_frame_time = 1 / self._video.fps

        if not self._video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self._video.path}')

        log.debug('waiting for first work request')
        self.worker_ready_event.wait()

        try:
            log.debug('starting task generation')
            self._stream_video()
        except Exception: # TODO find correct exception
            log.debug(f'Exception while streaming video file: {self._video.path}')
        finally:
            self.stop()

    def stop(self):
        self._video.close()

    def _stream_video(self):
        while self._video.is_opened():
            ok = self._iteration()
            if not ok:
                break

    def _iteration(self):
        iteration_start_time = time.perf_counter()

        ret, frame, frame_index = self._video.read_frame()
        if not ret:
            log.error("End of video stream or error reading frame.")
            return False

        self._add_to_queue(frame_index, frame)

        self._enforce_target_fps(iteration_start_time, self._target_frame_time)

        return True

    def _add_to_queue(self, task_id: int, data):
        self._queue.put(Task(task_id, TaskType.INFERENCE ,data))

    @staticmethod
    def _enforce_target_fps(iteration_start_time: float, target_frame_interval: float):
        iteration_duration = time.perf_counter() - iteration_start_time
        wait_time = max(target_frame_interval - iteration_duration, 0)
        if wait_time > 0:
            time.sleep(wait_time)