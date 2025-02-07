from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    id: int
    type: str
    data: np.ndarray