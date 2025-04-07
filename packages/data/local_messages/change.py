from dataclasses import dataclass

from packages.data.local_messages.local_message import LocalMessage


@dataclass(frozen=True)
class Change(LocalMessage):
    value: int