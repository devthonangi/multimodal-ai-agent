from collections import OrderedDict
from threading import RLock


class CacheManager:
    """Small thread-safe LRU cache for repeated image questions."""

    def __init__(self, max_items=128):
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self.max_items = max_items
        self._cache = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key, value):
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        with self._lock:
            return len(self._cache)

    def get_or_compute(self, key, compute_fn):
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.put(key, result)
        return result
