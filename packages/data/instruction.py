from dataclasses import dataclass


@dataclass
class Instruction:
    type: str
    value: int