from threading import Lock


class ConcurrentDict:
    def __init__(self):
        self._dict = {}
        self._lock = Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def pop(self, key, default=None):
        with self._lock:
            return self._dict.pop(key, default)