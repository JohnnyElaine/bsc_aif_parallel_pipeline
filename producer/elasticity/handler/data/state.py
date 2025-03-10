from dataclasses import dataclass


@dataclass
class State:
    current_index: int
    possible_states: list

    @property
    def value(self):
        return self.possible_states[self.current_index]