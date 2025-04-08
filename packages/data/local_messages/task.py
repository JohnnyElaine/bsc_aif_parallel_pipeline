import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Task:
    type: str = field(compare=False)
    id: int
    data: np.ndarray = field(compare=False)