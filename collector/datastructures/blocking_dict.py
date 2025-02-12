from threading import Lock, Semaphore


class BlockingDict:
    def __init__(self):
        self._dict = {}
        self._lock = Lock()
        self._items_available = Semaphore(0)

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

        # Signal that a new item is available
        self._items_available.release()

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
        # Signal that a new item is available
        self._items_available.release()

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def pop(self, key, default=None):
        # Wait for an item to become available
        self._items_available.acquire()
        with self._lock:
            return self._dict.pop(key, default)
