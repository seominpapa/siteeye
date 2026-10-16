"""Low-latency frame source: RTSP/HTTP stream, video file or webcam.

A reader thread always keeps only the newest frame, so a slow detector drops
stale frames instead of building up delay. Dead streams are reopened.
"""
import os
import threading
import time

import cv2

# Read by FFmpeg when a capture is opened. These options only suit RTSP: files fail to decode with them.
RTSP_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class FrameSource:
    def __init__(self, src, reconnect_s=2.0, pace_files=True, loop=False):
        self.src = int(src) if str(src).isdigit() else src
        self.reconnect_s = reconnect_s
        self.loop = loop
        self.is_file = isinstance(self.src, str) and os.path.isfile(self.src)
        self.pace = self.is_file and pace_files  # play files at native fps like a live camera
        self.fps = 0.0
        self.size = (0, 0)
        self.ended = False
        self._cond = threading.Condition()
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()

    def read(self, last_seq=0, timeout=5.0):
        """Block until a frame newer than last_seq arrives. Returns (seq, ts, frame) or None."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq and not self.ended and not self._stop:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
            if self._seq <= last_seq:
                return None
            return self._seq, self._ts, self._frame

    def _publish(self, frame):
        with self._cond:
            self._frame, self._ts = frame, time.time()
            self._seq += 1
            self._cond.notify_all()

    def _run(self):
        while not self._stop:
            if isinstance(self.src, str) and self.src.lower().startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_OPTIONS
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                time.sleep(self.reconnect_s)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            interval = 1.0 / self.fps if self.pace else 0.0
            nxt = time.monotonic()
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                self._publish(frame)
                if interval:
                    nxt += interval
                    time.sleep(max(0.0, nxt - time.monotonic()))
            cap.release()
            if self.is_file and not self.loop:
                with self._cond:
                    self.ended = True
                    self._cond.notify_all()
                return
            if not self.is_file:
                time.sleep(self.reconnect_s)
