from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    id: int
    type: str
    data: np.ndarray