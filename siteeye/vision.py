"""Hybrid vision stack: three complementary tracks merged into one detection list.

Track A  yolo11m-pose + ByteTrack      persons: stable track ids + 17 keypoints (posture, gaze, gait)
Track B  YOLO-World (zero-shot)        construction vocabulary with no fine-tuning (PPE, machinery, ladders)
Track C  custom_ppe.pt (fine-tuned)    small PPE (Hardhat / No-Hardhat / Safety-Hook) when a site model exists

Tracks that cannot load (missing weights, missing CLIP for YOLO-World, ...) are skipped with a warning
so the pipeline keeps running on whatever is available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .schemas import BBox, Detection, Keypoint

log = logging.getLogger("siteeye.vision")

_ALIASES = {
    "hardhat": "hard hat", "helmet": "hard hat", "hard hat": "hard hat",
    "no hardhat": "no hard hat", "no helmet": "no hard hat", "no hard hat": "no hard hat",
    "safety vest": "safety vest", "vest": "safety vest", "no safety vest": "no safety vest",
    "safety hook": "safety hook", "hook": "safety hook",
    "safety harness lanyard": "harness lanyard", "harness lanyard": "harness lanyard", "lanyard": "harness lanyard",
    # class names used by common construction-safety datasets (Roboflow / Hugging Face PPE models)
    "machinery": "heavy machinery", "heavy machinery": "heavy machinery", "heavy equipment": "heavy machinery",
    "heavy construction vehicle": "heavy machinery", "construction machine": "heavy machinery",
    "yellow hard hat": "hard hat",
    "safety helmet": "hard hat", "no safety helmet": "no hard hat", "no vest": "no safety vest",
}


def canonical_label(name: str) -> str:
    """Map class names from any track (incl. Roboflow-style 'No-Hardhat') to one shared vocabulary."""
    n = name.strip().lower().replace("-", " ").replace("_", " ")
    return _ALIASES.get(n, n)


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_weights(name: str, models_dir: Path) -> str | None:
    """Resolve a weights file: explicit path, else models_dir/<name>, else download an Ultralytics asset there."""
    p = Path(name)
    if p.exists():
        return str(p)
    target = models_dir / p.name
    if target.exists():
        return str(target)
    if not p.name.lower().startswith("yolo"):  # custom weights are never downloadable
        return None
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        models_dir.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(target))
    except Exception as exc:  # noqa: BLE001 - offline / unknown asset
        log.warning("could not fetch %s: %s", p.name, exc)
    return str(target) if target.exists() else None


class VisionTrack(Protocol):
    name: str

    def __call__(self, frame: np.ndarray) -> list[Detection]: ...


def _to_detections(result, source: str, with_ids: bool = False, with_kpts: bool = False) -> list[Detection]:
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.int().cpu().tolist()
    conf = b.conf.cpu().tolist()
    ids = b.id.int().cpu().tolist() if with_ids and b.id is not None else [None] * len(cls)
    kpts = result.keypoints.data.cpu().numpy() if with_kpts and result.keypoints is not None else None
    out = []
    for i, (x, c, p) in enumerate(zip(xyxy, cls, conf)):
        kp = [Keypoint(x=float(k[0]), y=float(k[1]), conf=float(k[2])) for k in kpts[i]] if kpts is not None else None
        out.append(Detection(label=canonical_label(result.names[c]), conf=float(p), source=source,  # type: ignore[arg-type]
                             box=BBox(x1=float(x[0]), y1=float(x[1]), x2=float(x[2]), y2=float(x[3])),
                             track_id=ids[i], keypoints=kp))
    return out


class PoseTrackA:
    name = "A"

    def __init__(self, weights: str, device: str, imgsz: int, conf: float):
        from ultralytics import YOLO

        self.model, self.device, self.imgsz, self.conf = YOLO(weights), device, imgsz, conf

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.conf,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return _to_detections(r, "A", with_ids=True, with_kpts=True)


class WorldTrackB:
    name = "B"

    def __init__(self, weights: str, prompts: list[str], device: str, imgsz: int, conf: float):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.model.set_classes(prompts)  # open-vocabulary prompts; needs the CLIP text encoder
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return _to_detections(r, "B")


class PPETrackC:
    name = "C"

    def __init__(self, weights: str, device: str, imgsz: int, conf: float):
        from ultralytics import YOLO

        self.model, self.device, self.imgsz, self.conf = YOLO(weights), device, imgsz, conf

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return _to_detections(r, "C")


class HybridVisionExtractor:
    """Runs track A every frame and the heavier aux tracks (B, C) every Nth frame, reusing their last output."""

    def __init__(self, tracks: dict[str, VisionTrack], aux_every: int, label_min_conf: dict[str, float] | None = None):
        self.tracks, self.aux_every = tracks, max(1, aux_every)
        self.label_min_conf = label_min_conf or {}
        self._aux_last: list[Detection] = []
        self.slot = tracks.get("A")  # the per-session pose model, so a pool can take it back after the session

    @property
    def active(self) -> str:
        return ",".join(self.tracks) or "none"

    def reset(self) -> None:
        """Start a new source: clear ByteTrack state so ids from the previous camera do not carry over."""
        self._aux_last = []
        a = self.tracks.get("A")
        pred = getattr(getattr(a, "model", None), "predictor", None)
        for t in getattr(pred, "trackers", []) or []:
            t.reset()

    def infer(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        dets: list[Detection] = []
        if "A" in self.tracks:
            dets += self._safe("A", frame)
        if frame_idx % self.aux_every == 0 or not self._aux_last:
            aux: list[Detection] = []
            for k in ("B", "C"):
                if k in self.tracks:
                    aux += self._safe(k, frame)
            self._aux_last = [d for d in aux if d.conf >= self.label_min_conf.get(d.label, 0.0)]
        # with track A active, persons come from A only (ids + keypoints); B/C persons would duplicate them
        keep_person = "A" not in self.tracks
        dets += [d for d in self._aux_last if keep_person or d.label != "person"]
        return dets

    def _safe(self, key: str, frame: np.ndarray) -> list[Detection]:
        try:
            return self.tracks[key](frame)
        except Exception as exc:  # noqa: BLE001 - one broken track must not stop the stream
            log.warning("track %s failed (%s); disabling it", key, exc)
            del self.tracks[key]
            return []


PRECISION_TRACKS = {"fast": "A", "accurate": "A,B", "full": "A,B,C"}  # web UI precision -> tracks


def weights_available(name: str, models_dir: Path) -> bool:
    """True if a weights file is already on disk (explicit path or inside models_dir)."""
    p = Path(name)
    return p.exists() or (models_dir / p.name).exists()


class VisionModelRegistry:
    """Factory: builds tracks and extractors. Tracks B and C are stateless, so one instance can serve many sessions;
    only track A (ByteTrack keeps its state inside the model) must be built per concurrent session."""

    @staticmethod
    def build_track(key: str, st: Settings, device: str) -> VisionTrack | None:
        try:
            if key == "A":
                w = ensure_weights(st.pose_weights, st.models_dir)
                return PoseTrackA(w, device, st.imgsz, st.conf_floor) if w else None
            if key == "B":
                w = ensure_weights(st.world_weights, st.models_dir)
                return WorldTrackB(w, st.world_prompts, device, st.imgsz, st.conf_floor) if w else None
            if key == "C":
                w = ensure_weights(st.ppe_weights, st.models_dir)  # custom weights: never downloaded, only used if present
                if w:
                    return PPETrackC(w, device, st.imgsz, st.conf_floor)
                log.warning("track C skipped: %s not found (put your PPE weights in %s)", st.ppe_weights, st.models_dir)
                return None
            log.warning("unknown track %r ignored", key)
        except Exception as exc:  # noqa: BLE001
            log.warning("track %s unavailable: %s", key, exc)
        return None

    @staticmethod
    def build(st: Settings) -> HybridVisionExtractor:
        """Extractor for the tracks listed in settings (SITEEYE_TRACKS=A,B,C), used by the CLI."""
        device = pick_device(st.device)
        wanted = [t.strip().upper() for t in st.tracks.split(",") if t.strip()]
        tracks: dict[str, VisionTrack] = {}
        for key in wanted:
            t = VisionModelRegistry.build_track(key, st, device)
            if t is not None:
                tracks[key] = t
        if not tracks:
            raise RuntimeError("no vision track could be loaded; check weights and SITEEYE_TRACKS")
        log.info("vision tracks active: %s on %s", ",".join(tracks), device)
        return HybridVisionExtractor(tracks, st.aux_every, st.label_min_conf)
