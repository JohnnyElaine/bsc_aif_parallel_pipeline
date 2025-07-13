import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Task:
    type: str = field(compare=False)
    id: int
    stream_key: int = field(compare=False)
    data: np.ndarray = field(compare=False)