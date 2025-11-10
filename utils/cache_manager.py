class CacheManager:
    def __init__(self):
        self.cache = {}

    def get_or_compute(self, key, compute_fn):
        if key in self.cache:
            print(f"[CACHE] Hit for {key}")
            return self.cache[key]
        print(f"[CACHE] Miss for {key}, computing...")
        result = compute_fn()
        self.cache[key] = result
        return result
