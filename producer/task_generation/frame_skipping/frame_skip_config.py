import math

class FrameSkipConfig:
    """
    config for skipping frames when target_fps < self._video.fps
    """
    def __init__(self, numerator=1, denominator=1, count=-1):
        self._numerator = numerator
        self._denominator = denominator
        self._count = count

    def set(self, numerator, denominator):
        gcd = math.gcd(numerator, denominator)
        self._numerator = numerator // gcd
        self._denominator = denominator // gcd

    def increment(self):
        self._count = (self._count + 1) % self._denominator

    def should_skip_frame(self) -> bool:
        """
        skip certain % of frames if current-fps < source-fps
        Returns: True if current-fps < source-fps, False otherwise
        """
        return self._count >= self._numerator