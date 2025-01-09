class OffloadingPreference:
    """
    Decides when to offload a frame:
    offload percentage = offload_frames / total_frames

    Example:
        offload_frames = 3
        total_frames = 4
        offload percentage = 3/4 = 75%
        75% of all frames that are being assigned to this node will be offloaded

    Inner function:
        counter is used to track at which frame of the current cycle it is at.
        when counter < offload_frames then it will decide to offload the frame

        So with 3/4=75% offloading, if counter=
        0 < 3 ==> offload
        1 < 3 ==> offload
        2 < 3 ==> offload
        3 < 3 ==> no offload
        (counter resets as it cannot hit value of total_frames)
        0 < 3 ==> offload
        1 < 3 ==> offload
        ...
    """
    def __init__(self, offload_frames: int, total_frames: int, targets):
        self.offload_frames = offload_frames
        self.total_frames = total_frames
        self.counter = 0  # tracks if current frame should be offloaded

        self.targets = targets

    def increment_and_decide(self):
        offload = False
        if self.counter < self.offload_frames:
            offload = True

        self.counter = (self.counter + 1) % self.total_frames

        return offload
