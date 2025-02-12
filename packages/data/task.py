from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, order=True)
class Task:
    id: int
    type: str = field(compare=False)
    data: np.ndarray = field(compare=False)