from collections import deque

class MovingAverage:
    def __init__(self, window_size=30):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def add(self, val: float):
        if len(self.window) == self.window.maxlen:
            self.sum -= self.window[0]

        self.window.append(val)
        self.sum += val

    def average(self):
        if len(self.window) == 0:
            return 0.0
        return self.sum / len(self.window)