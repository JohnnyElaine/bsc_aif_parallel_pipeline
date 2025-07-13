import logging
import time
from queue import Queue
from threading import Thread
from collections import deque
import numpy as np

import packages.time_util as time_util
from packages.data import Task, TaskType
from packages.data.video.video import Video
from producer.data.resolution import Resolution
from producer.data.stream_multiplier_threshold import StreamMultiplierThreshold
from producer.task_generation.frame_skipping.frame_skip_config import FrameSkipConfig

log = logging.getLogger("producer")


class TaskGenerator(Thread):

    MAX_QUEUE_SIZE = 120

    def __init__(self, shared_queue: Queue, video: Video, start_event, stream_multiplier_schedule=None):
        super().__init__()
        self._queue = shared_queue
        self._video = video
        self._target_resolution = self._video.resolution
        self._target_fps = self._video.fps
        self._target_frame_time = 1 / self._target_fps
        self._start_event = start_event

        self._task_id = 0

        self._time_last_frame_generated_at = 0

        self._frame_skip_config = FrameSkipConfig()

        # Only used for Evaluation:
        # Stream multiplier for simulating multiple streams
        self._stream_multiplier = 1

        # Schedule for changing stream multiplier: list of StreamMultiplierEntry objects
        self._total_frames = self._video.frame_count
        
        # Pre-calculate frame thresholds for efficiency (assume schedule is already sorted)
        self._next_schedule_threshold = None
        if stream_multiplier_schedule is not None and len(stream_multiplier_schedule) > 0:
            self._schedule_thresholds = deque()
            for entry in stream_multiplier_schedule:
                self._schedule_thresholds.append(StreamMultiplierThreshold(int(self._total_frames * entry.frame_percentage), entry.multiplier))
            self._next_schedule_threshold = self._schedule_thresholds.popleft()

    def run(self):
        if not self._video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self._video.path}')

        log.debug('waiting for first work request')
        self._start_event.wait()

        try:
            log.debug('starting task generation')
            self._stream_video()
        except MemoryError:
            # Handle memory-related errors
            log.error("Out of memory while processing video.")
        except OSError as e:
            # Handle file I/O or system-related errors
            log.error(f'System error while streaming video file {self._video.path}: {e}')
        except Exception as e:
            log.error(f"OpenCV error while streaming video file {self._video.path}: {e}")
        finally:
            self._stop_request_handler()
            self._video.close()
        log.debug('stopped task-generator')

    def queue_size(self):
        """Return the approximate size of the queue (not reliable!)."""
        return self._queue.qsize()

    def set_fps(self, fps: int):
        self._target_fps = min(fps, self._video.fps)
        self._target_frame_time = 1 / self._target_fps

        self._frame_skip_config.set(self._target_fps, self._video.fps)

    def set_resolution(self, res: Resolution):
        self._target_resolution = res

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

        self._update_stream_multiplier()

        self._frame_skip_config.increment()

        if self._frame_skip_config.should_skip_frame():
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

    def _update_stream_multiplier(self):
        if self._next_schedule_threshold is None:
            return

        if self._video.frame_index < self._next_schedule_threshold.frame:
            return

        self._stream_multiplier = self._next_schedule_threshold.multiplier
        
        # Move to next threshold if available
        if self._schedule_thresholds:
            self._next_schedule_threshold = self._schedule_thresholds.popleft()
        else:
            self._next_schedule_threshold = None

    def _add_to_queue(self, data: np.ndarray):
        """
        Add the data to the task_queue (frame buffer)
        Add n=self._stream_multiplier copies to simulate multiple streams
        n=1, 1 active stream
        n=3, 1 real stream and 2 copies, simulating 3 active streams
        Args:
            data:
        """
        for stream_id in range(self._stream_multiplier):
            frame_copy = data.copy() if stream_id > 0 else data
            self._queue.put(Task(TaskType.INFERENCE, self._task_id, stream_id, frame_copy))
            self._task_id += 1

    def _stop_request_handler(self):
        self._queue.put(Task(TaskType.END, -1, 0, np.empty(0)))


    @property
    def fps(self) -> int:
        return self._target_fps

    @property
    def frame_time(self) -> float:
        return self._target_frame_time