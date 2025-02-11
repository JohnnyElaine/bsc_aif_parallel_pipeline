from dataclasses import dataclass


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int