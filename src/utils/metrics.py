from collections import deque


class Meter:
    def __init__(self, maxlen: int = 200):
        self.count = 0
        self._lat = deque(maxlen=maxlen)

    def observe(self, seconds: float):
        self.count += 1
        self._lat.append(seconds)

    def summary(self):
        arr = list(self._lat)
        if not arr:
            return {"count": self.count, "p50_ms": 0, "p95_ms": 0}
        arr.sort()

        def pct(p):
            # nearest-rank percentile
            if len(arr) == 1:
                return arr[0]
            k = max(0, min(len(arr) - 1, round((p / 100.0) * (len(arr) - 1))))
            return arr[int(k)]

        return {
            "count": self.count,
            "p50_ms": round(pct(50) * 1000, 2),
            "p95_ms": round(pct(95) * 1000, 2),
        }
