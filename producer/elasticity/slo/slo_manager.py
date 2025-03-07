from producer.data.resolution import Resolution
from producer.elasticity.elasticity_handler import ElasticityHandler


class SloManager:
    def __init__(self, elasticity_handler: ElasticityHandler,
                 target_fps: int,
                 target_resolution: Resolution,
                 fps_tolerance = 0.8,
                 resolution_tolerance = 0.8):
        self.elasticity_handler = elasticity_handler

        self.target_fps = target_fps
        self.target_resolution = target_resolution

        self.fps_tolerance = fps_tolerance
        self.resolution_tolerance = resolution_tolerance

    def queue_slo_satisfied(self):
        max_qsize = 2 * self.elasticity_handler.fps # Example: 2 seconds worth of frames
        return self.elasticity_handler.queue_size() <= max_qsize

    def fps_slo_satisfied(self):
        return self.elasticity_handler.fps >= self.target_fps * self.fps_tolerance

    def resolution_slo_satisfied(self):
        return self.elasticity_handler.resolution.pixels >= self.target_resolution.pixels * self.resolution_tolerance
