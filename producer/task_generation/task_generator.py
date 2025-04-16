import logging
import math
import time
from queue import Queue
from threading import Thread

import numpy as np
from cv2 import error as cvError

import packages.time_util as time_util
from packages.data import Task, TaskType
from producer.data.resolution import Resolution
from packages.data.video.video import Video

log = logging.getLogger("producer")


class TaskGenerator(Thread):

    MAX_QUEUE_SIZE = 150

    def __init__(self, shared_queue: Queue, video: Video, start_event):
        super().__init__()
        self._queue = shared_queue
        self._video = video
        self._target_resolution = self._video.resolution
        self._target_frame_time = 1 / self._video.fps
        self._start_event = start_event

        self._task_id = 0

        self._time_last_frame_generated_at = 0

        # values for skipping frames when target_fps < self._video.fps
        self._numerator = 1
        self._denominator = 1
        self._count = -1

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
            log.error(f'System error while streaming video file {self._video.path}: {e}')
        finally:
            self._stop_request_handler()
            self._video.close()
        log.debug('stopped task-generator')

    def queue_size(self):
        """Return the approximate size of the queue (not reliable!)."""
        return self._queue.qsize()

    def set_fps(self, fps: int):
        self._target_frame_time = min(1 / fps, self._video.fps)

        self._numerator = fps
        self._denominator = self._video.fps

        gcd = math.gcd(self._numerator, self._denominator)
        self._numerator //= gcd
        self._denominator //= gcd

    def _stream_video(self):
        ok = True
        self._time_last_frame_generated_at = time.perf_counter() # init before first iteration
        while self._video.is_opened() and ok:
            ok = self._iteration()

    def _iteration(self) -> bool:
        grabbed, frame_index = self._video.grab()

        if not grabbed:
            log.error("End of video stream or error grabbing frame.")
            return False

        self._count = (self._count + 1) % self._denominator

        # skip frame certain frame if current-fps < source-fps
        if self._count >= self._numerator:
            return True

        ret, frame = self._video.retrieve()

        if not ret:
            log.error("Error retrieving frame")
            return False

        if self._target_resolution != self._video.resolution:
            frame = self._video.resize_frame(frame, self._target_resolution.width, self._target_resolution.height)

        self._add_to_queue(frame)

        # Enforce timing based on target FPS
        time_util.enforce_target_fps(self._time_last_frame_generated_at, self._target_frame_time)

        # track when last frame/task was generated
        self._time_last_frame_generated_at = time.perf_counter()

        return True

    def _add_to_queue(self, data: np.ndarray):
        self._queue.put(Task(TaskType.INFERENCE, self._task_id ,data))
        self._task_id += 1

    def _stop_request_handler(self):
        self._queue.put(Task(TaskType.END, -1, np.empty(0)))

    def _is_resolution_changed(self):
        return self._target_resolution[0]