"""Turn per-frame detections into scene state and enter/exit/dwell events."""
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Event:
    kind: str  # "enter" | "exit" | "dwell"
    track_id: int
    name: str
    t: float
    in_zone: bool = False
    dwell_s: float = 0.0


@dataclass
class _Track:
    name: str
    first: float
    last: float
    hits: int = 0
    confirmed: bool = False
    in_zone: bool = False
    zone_since: float = 0.0
    dwell_fired: bool = False


class Zone:
    """Polygon in normalized (0-1) image coordinates; a box is 'in' when its bottom-center is."""

    def __init__(self, points):
        self.points = np.array(points, dtype=np.float32)

    @classmethod
    def parse(cls, text):
        # "0.1,0.5;0.9,0.5;0.9,1;0.1,1"
        return cls([tuple(float(v) for v in p.split(",")) for p in text.split(";")])

    def contains(self, xyxy, frame_wh):
        w, h = frame_wh
        x, y = (xyxy[0] + xyxy[2]) / 2, xyxy[3]
        poly = (self.points * np.array([w, h], dtype=np.float32)).astype(np.float32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def pixels(self, frame_wh):
        return (self.points * np.array(frame_wh, dtype=np.float32)).astype(np.int32)


@dataclass
class SceneTracker:
    zone: Zone = None
    min_hits: int = 3  # frames before a track counts (filters flicker)
    lost_after_s: float = 1.5  # silence before a track is considered gone
    dwell_s: float = 10.0  # time in zone before a dwell event
    tracks: dict = field(default_factory=dict)

    def update(self, dets, t, frame_wh):
        events = []
        for d in dets:
            if d.track_id < 0:
                continue
            tr = self.tracks.get(d.track_id)
            if tr is None:
                tr = self.tracks[d.track_id] = _Track(d.name, t, t)
            tr.last, tr.hits = t, tr.hits + 1
            inside = self.zone.contains(d.xyxy, frame_wh) if self.zone else False
            if inside and not tr.in_zone:
                tr.zone_since = t
            tr.in_zone = inside
            if not tr.confirmed and tr.hits >= self.min_hits:
                tr.confirmed = True
                events.append(Event("enter", d.track_id, tr.name, t, inside))
            if tr.confirmed and inside and not tr.dwell_fired and t - tr.zone_since >= self.dwell_s:
                tr.dwell_fired = True
                events.append(Event("dwell", d.track_id, tr.name, t, True, t - tr.zone_since))
        for tid in [k for k, v in self.tracks.items() if t - v.last > self.lost_after_s]:
            tr = self.tracks.pop(tid)
            if tr.confirmed:
                events.append(Event("exit", tid, tr.name, t, tr.in_zone, tr.last - tr.first))
        return events

    def state(self, now=None):
        live = {k: v for k, v in self.tracks.items() if v.confirmed}
        return {
            "counts": dict(Counter(v.name for v in live.values())),
            "in_zone": dict(Counter(v.name for v in live.values() if v.in_zone)),
            "tracks": [
                {
                    "id": k,
                    "name": v.name,
                    "in_zone": v.in_zone,
                    "age_s": (now or v.last) - v.first,
                    "zone_s": ((now or v.last) - v.zone_since) if v.in_zone else 0.0,
                }
                for k, v in sorted(live.items())
            ],
        }
