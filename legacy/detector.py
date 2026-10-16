"""YOLO detection + multi-object tracking (ByteTrack) for one frame at a time."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    track_id: int  # -1 when the tracker has not assigned an id yet
    cls: int
    name: str
    conf: float
    xyxy: tuple


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, weights="yolo11n.pt", imgsz=640, conf=0.35, classes=None, device=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.conf = imgsz, conf
        self.device = device or pick_device()
        names = self.model.names
        self.names = names
        # classes are given by name ("person car") and mapped to ids
        self.class_ids = None
        if classes:
            by_name = {v: k for k, v in names.items()}
            unknown = [c for c in classes if c not in by_name]
            if unknown:
                raise ValueError(f"unknown classes {unknown}; model knows {sorted(by_name)}")
            self.class_ids = [by_name[c] for c in classes]
        # the first inference compiles kernels (seconds on MPS/CUDA); pay that before the stream starts
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=self.device, verbose=False)

    def __call__(self, frame):
        r = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.int().cpu().tolist()
        conf = b.conf.cpu().tolist()
        ids = b.id.int().cpu().tolist() if b.id is not None else [-1] * len(cls)
        return [
            Detection(i, c, self.names[c], p, tuple(map(float, x)))
            for i, c, p, x in zip(ids, cls, conf, xyxy)
        ]
