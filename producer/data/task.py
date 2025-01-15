from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    id: int
    task: np.ndarray