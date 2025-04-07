import numpy as np
from dataclasses import dataclass, field


from packages.data.local_messages.local_message import LocalMessage


@dataclass(frozen=True, order=True)
class Task(LocalMessage):
    id: int
    data: np.ndarray = field(compare=False)