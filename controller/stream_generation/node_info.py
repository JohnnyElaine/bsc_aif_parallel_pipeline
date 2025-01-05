class NodeInfo:
    def __init__(self, id: int):
        self.id = id
        self.offloading_preference = None

    def is_offloading(self):
        if self.offloading_preference is None:
            return False

        return self.offloading_preference.increment_and_decide()