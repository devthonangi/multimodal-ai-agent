import time

import cv2


class VideoStream:
    def __init__(self, src=0, fps=1):
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        self.interval = 1.0 / fps

    def frames(self):
        last = 0.0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                now = time.monotonic()
                if now - last >= self.interval:
                    last = now
                    yield frame
        finally:
            self.cap.release()
