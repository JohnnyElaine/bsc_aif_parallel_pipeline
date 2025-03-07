import logging
import time
from threading import Thread
from queue import Queue
from cv2 import error as cvError

import packages.time_util as time_util
from packages.data import Task, TaskType
from producer.data.resolution import Resolution
from producer.data.video import Video

log = logging.getLogger("producer")


class TaskGenerator(Thread):
    MAX_QUEUE_SIZE = 200
    def __init__(self, shared_queue: Queue, video: Video, start_event):
        super().__init__()
        self._queue = shared_queue
        self._video = video
        self._target_resolution = self._video.resolution
        self._target_frame_time = 1 / self._video.fps
        self._start_event = start_event

    def run(self):
        if not self._video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self._video.path}')

        log.debug('waiting for first work request')
        self._start_event.wait()

        try:
            log.debug('starting task generation')
            self._stream_video()
        except cvError as e:
            # Handle OpenCV-specific errors (e.g., video file issues, frame processing errors)
            log.error(f"OpenCV error while streaming video file {self._video.path}: {e}")
        except MemoryError:
            # Handle memory-related errors
            log.error("Out of memory while processing video.")
        except OSError as e:
            # Handle file I/O or system-related errors
            log.error(f"System error while streaming video file {self._video.path}: {e}")
        finally:
            self.stop()

    def stop(self):
        self._video.close()

    def queue_size(self):
        """Return the approximate size of the queue (not reliable!)."""
        return self._queue.qsize()

    def set_fps(self, fps: int):
        self._target_frame_time = min(1 / fps, self._video.fps)

    def set_resolution(self, res: Resolution):
        self._target_resolution = res

    def _stream_video(self):
        ok = True
        while self._video.is_opened() and ok:
            ok = self._iteration()

    def _iteration(self) -> bool:
        iteration_start_time = time.perf_counter()

        ret, frame, frame_index = self._video.read_frame()
        if not ret:
            log.error("End of video stream or error reading frame.")
            return False

        if self._target_resolution != self._video.resolution:
            frame = self._video.resize_frame(frame, self._target_resolution.width, self._target_resolution.height)

        self._add_to_queue(frame_index, frame)

        time_util.enforce_target_fps(iteration_start_time, self._target_frame_time)

        return True

    def _add_to_queue(self, task_id: int, data):
        self._queue.put(Task(task_id, TaskType.INFERENCE ,data))

    def _is_resolution_changed(self):
        return self._target_resolution[0]