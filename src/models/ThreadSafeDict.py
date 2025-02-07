import threading

class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def get(self, key):
        with self._lock:
            return self._dict.get(key)

    def remove(self, key):
        with self._lock:
            if key in self._dict:
                del self._dict[key]

    def __repr__(self):
        with self._lock:
            return repr(self._dict)
    
    def contains(self, key):
        """Check if a key is in the dictionary in a thread-safe manner."""
        with self._lock:
            return key in self._dict