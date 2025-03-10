from dataclasses import dataclass


@dataclass(frozen=True)
class Instruction:
    type: str
    value: int