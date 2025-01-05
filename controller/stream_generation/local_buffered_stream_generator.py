import logging
import time

from controller.stream_generation.node_info import NodeInfo
from shared.shared_frame_buffer import SharedFrameBuffer
from controller.stream_generation.video import Video

log = logging.getLogger("controller")

class LocalBufferedStreamGenerator:
    def __init__(self, shared_frame_buffer: SharedFrameBuffer, nodes: list[NodeInfo], video_path):
        self._shared_frame_buffer = shared_frame_buffer
        self.nodes = nodes
        self._video = Video(video_path)
        self._target_frame_time = 1 / self._video.fps
        self._curr_target = 0
        self._curr_offload_target = 0

    def start(self):
        if not self._video.is_opened():
            raise IOError(f'Unable to open input video file. Path: {self._video.path}')

        self._play_video()
        self.stop()

    def stop(self):
        """
        Stops the video stream, releases the video capture
        :return:
        """
        if not self._video.is_opened():
            return

        self._video.release()

    def _play_video(self):
        while self._video.is_opened():
            iteration_start_time = time.time()

            ret, frame = self._video.read_frame()
            if not ret:
                log.debug("End of video stream or error reading frame.")
                break

            self._add_frame_to_buffer(frame)
            self._enforce_target_fps(iteration_start_time)

    def _add_frame_to_buffer(self, frame):
        index = self._handle_offloading()

        self._shared_frame_buffer.add(frame, index)
        self._curr_target = (self._curr_target + 1) % len(self.nodes)

    def _handle_offloading(self):
        if not self.nodes[self._curr_target].is_offloading():
            return self._curr_target

        self._increment_offload_target()

        # prevent offloading back to the node that requested offloading
        if self._curr_offload_target == self._curr_target:
            self._increment_offload_target()

        return self._curr_offload_target

    def _increment_offload_target(self):
        self._curr_offload_target = (self._curr_offload_target + 1) % len(self.nodes)

    def _enforce_target_fps(self, iteration_start_time: float):
        iteration_duration = time.time() - iteration_start_time
        wait_time = max(self._target_frame_time - iteration_duration, 0)
        if wait_time > 0:
            time.sleep(wait_time)
