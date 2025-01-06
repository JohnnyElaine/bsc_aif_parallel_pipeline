from abc import ABC

from controller.stream_generation.stream_generator import StreamGenerator


class LocalStreamGenerator(StreamGenerator, ABC):
    def __init__(self, video_path):
        super().__init__(video_path)