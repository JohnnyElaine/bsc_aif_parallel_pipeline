class State:
    """
    lower index = less quality
    higher index = higher quality
    """
    
    def __init__(self, starting_index:int, possible_states:list, change_function: callable):
        self.current_index = starting_index
        self.possible_states = possible_states
        self.change_function = change_function

    @property
    def value(self):
        return self.possible_states[self.current_index]

    def max(self):
        return self.possible_states[len(self.possible_states) - 1]

    def change(self, index: int):
        if index == self.current_index:
            return False
        if index < 0 or index >= len(self.possible_states):
            return False

        self.current_index = index

        return self.change_function(self.value)

    def capacity(self):
        """
        Calculates and returns the current capacity for this state

        Returns:
            The current capacity (ratio) in the range (0-1)
        """
        return self.current_index / (len(self.possible_states) - 1)

    def can_increase(self) -> bool:
        return self.current_index < len(self.possible_states) - 1

    def can_decrease(self) -> bool:
        return self.current_index > 0

    def increase(self):
        if self.current_index >= len(self.possible_states) - 1:
            return False

        self.current_index += 1
        self.change_function(self.value)
        return True
    
    def decrease(self):
        if self.current_index <= 0:
            return False

        self.current_index -= 1
        self.change_function(self.value)

        return True
    

    
    