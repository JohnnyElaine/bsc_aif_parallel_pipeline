class BufferedStreamSimulator:
    def __init__(self):
        pass


# Idea
# Start as separate thread
# read from .mp4 file and simulate 30fps
# write to a shared memory that has:
# a buffer (queue) for every node where frames are placed into
# the node then consumes the oldest frame from the buffer (queue)
