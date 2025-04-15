from dataclasses import dataclass


@dataclass
class State:
    """
    lower index = less quality
    higher index = high quality
    """
    current_index: int
    possible_states: list

    @property
    def value(self):
        return self.possible_states[self.current_index]

    @property
    def max(self):
        return self.possible_states[len(self.possible_states) - 1]

    def can_increase(self) -> bool:
        return self.current_index < len(self.possible_states) - 1

    def can_decrease(self) -> bool:
        return self.current_index > 0