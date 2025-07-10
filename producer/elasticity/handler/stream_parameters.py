from producer.elasticity.handler.data.state import State


class StreamParameters:
    def __init__(self, state_resolution: State, state_fps: State, state_inference_quality: State):
        self.state_resolution = state_resolution
        self.state_fps = state_fps
        self.state_inference_quality = state_inference_quality
