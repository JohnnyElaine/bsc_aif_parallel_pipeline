from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    @property
    def pixels(self):
        return self.width * self.height

    def get_aspect_ratio(self):
        common_divisor = gcd(self.width, self.height)
        simplified_width = self.width // common_divisor
        simplified_height = self.height // common_divisor

        return Resolution(simplified_width, simplified_height)