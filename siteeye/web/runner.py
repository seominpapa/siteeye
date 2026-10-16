"""AnalysisRunner: one video source -> vision -> SafetyPipeline -> every connected browser.

Wire format on /ws/stream
  binary  = uint32(meta_len) | meta JSON | JPEG      (latest frame only; slow browsers skip frames)
  text    = {"type": "status" | "event" | "history", ...}   (reliable: decisions and actions are never dropped)
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel

from ..config import Settings
from ..jev_client import JevSafetyPredictor, RoiImage
from ..mock import simulated_transport
from ..pipeline import SafetyPipeline
from ..router import Action, SafetyActionRouter
from ..schemas import BBox, FrameFeatures, JevDecision, RiskAssessment, SequencePayload
from ..source import FrameSource
from ..spatial import SpatialFeatureProcessor
from ..vision import HybridVisionExtractor, VisionModelRegistry
from .demo import DEMOS, ScriptedDemo

log = logging.getLogger("siteeye.web")
SEND_FPS, SEND_MAX_W = 15, 960
ACTION_KO = {
    "equipment_estop": "장비 비상정지 인터록 발동",
    "speaker_alert": "현장 스피커 경고방송",
    "smartwatch_vibration": "작업자 스마트워치 진동 경고",
    "dashboard_review": "안전관리자 확인 요청 (스냅샷·시퀀스 전송)",
    "monitor": "정상 모니터링",
}
PLAN_KO = {
    "normal_monitoring": "정상 모니터링", "worker_smartwatch_vibration": "스마트워치 진동 경고",
    "site_speaker_announcement": "현장 스피커 경고방송", "equipment_emergency_stop_interlock": "장비 비상정지 인터록",
}


class StartRequest(BaseModel):
    kind: Literal["demo", "url", "webcam", "browser"]
    demo_id: str | None = None
    url: str | None = None
    index: int = 0
    precision: Literal["fast", "accurate"] = "fast"
    loop: bool = True


class _Client:
    """Per-browser sender. Frames are latest-wins; control messages are queued and always delivered."""

    def __init__(self, ws: WebSocket):
        self.ws, self.frame, self.ctrl, self.ev = ws, None, [], asyncio.Event()
        self.task = asyncio.create_task(self._pump())

    def send_frame(self, data: bytes) -> None:
        self.frame = data
        self.ev.set()

    def send_ctrl(self, msg: dict) -> None:
        self.ctrl.append(json.dumps(msg, ensure_ascii=False))
        self.ev.set()

    async def _pump(self) -> None:
        try:
            while True:
                await self.ev.wait()
                self.ev.clear()
                ctrl, self.ctrl = self.ctrl, []
                for m in ctrl:
                    await self.ws.send_text(m)
                if self.frame is not None:
                    data, self.frame = self.frame, None
                    await self.ws.send_bytes(data)
        except Exception:  # noqa: BLE001 - browser went away
            pass

    def close(self) -> None:
        self.task.cancel()


def _norm(b: BBox, w: int, h: int) -> list[float]:
    return [round(b.x1 / w, 4), round(b.y1 / h, 4), round(b.x2 / w, 4), round(b.y2 / h, 4)]


class AnalysisRunner:
    def __init__(self, st: Settings, offline: bool, out_dir: Path):
        self.st, self.offline = st, offline
        self.live = bool(st.api_key) and not offline
        self.snap_dir = out_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.clients: set[_Client] = set()
        self.history: deque[dict] = deque(maxlen=150)
        self.status: dict = {"state": "idle", "message": "분석을 시작하세요"}
        self._task: asyncio.Task | None = None
        self._pipe: SafetyPipeline | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)  # single inference thread
        self._extractors: dict[str, HybridVisionExtractor] = {}
        self._ingest: asyncio.Queue | None = None
        self._wh, self._last_jpeg, self._demo_zones = (1, 1), b"", {}

    # ------------------------------------------------------------------ clients and messages

    def add_client(self, ws: WebSocket) -> _Client:
        c = _Client(ws)
        self.clients.add(c)
        c.send_ctrl({"type": "status", **self.status})
        c.send_ctrl({"type": "history", "events": list(self.history)})
        return c

    def remove_client(self, c: _Client) -> None:
        self.clients.discard(c)
        c.close()

    def _set_status(self, state: str, message: str = "", **extra) -> None:
        self.status = {"state": state, "message": message, **extra}
        for c in self.clients:
            c.send_ctrl({"type": "status", **self.status})

    def _emit(self, ev: dict) -> None:
        self.history.append(ev)
        for c in self.clients:
            c.send_ctrl({"type": "event", **ev})

    # ------------------------------------------------------------------ lifecycle

    async def start(self, req: StartRequest) -> None:
        await self.stop()
        if req.kind == "demo" and req.demo_id not in DEMOS:
            raise ValueError("unknown demo")
        if req.kind == "url" and not (req.url or "").lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise ValueError("URL must start with rtsp://, rtsps://, http:// or https://")
        self.history.clear()
        self._set_status("starting", "시작하는 중…")
        self._task = asyncio.create_task(self._run(req))

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
        self._set_status("idle", "중지됨")

    def push_ingest(self, data: bytes) -> None:
        q = self._ingest
        if q is None:
            return
        if q.full():
            q.get_nowait()  # drop the oldest frame, keep latency low
        q.put_nowait(data)

    # ------------------------------------------------------------------ sources

    async def _capture(self, src: FrameSource):
        loop = asyncio.get_running_loop()
        seq, waiting = 0, False
        src.start()
        try:
            while True:
                got = await loop.run_in_executor(None, src.read, seq)
                if got is None:
                    if src.ended:
                        return
                    if not waiting:
                        waiting = True
                        self._set_status("connecting", "영상 신호를 기다리는 중… (주소·권한을 확인하세요)")
                    continue
                if waiting:
                    waiting = False
                    self._set_status("running", "분석 중")
                seq, ts, frame = got
                yield ts, frame, None, False
        finally:
            src.stop()

    async def _browser_frames(self):
        self._ingest = asyncio.Queue(maxsize=2)
        self._set_status("connecting", "브라우저 웹캠 연결을 기다리는 중…")
        got_first = False
        while True:
            try:
                data = await asyncio.wait_for(self._ingest.get(), timeout=10)
            except asyncio.TimeoutError:
                got_first = False
                self._set_status("connecting", "웹캠 영상이 들어오지 않습니다")
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if not got_first:
                got_first = True
                self._set_status("running", "분석 중")
            yield time.time(), img, None, False

    def _extractor(self, precision: str) -> HybridVisionExtractor:
        tracks = "A" if precision == "fast" else "A,B"
        if tracks not in self._extractors:
            self._extractors[tracks] = VisionModelRegistry.build(self.st.model_copy(update={"tracks": tracks}))
        ex = self._extractors[tracks]
        ex.reset()
        return ex

    # ------------------------------------------------------------------ main loop

    async def _run(self, req: StartRequest) -> None:
        loop = asyncio.get_running_loop()
        client = None if self.live else httpx.AsyncClient(transport=simulated_transport())
        predictor = JevSafetyPredictor(self.st, client)
        pipe = self._pipe = SafetyPipeline(self.st, predictor, SafetyActionRouter(self.st, sinks=[]), verbose=False)
        pipe.on_result = self._on_result
        source, gen, extractor, proc = None, None, None, None
        self._demo_zones = {}
        try:
            if req.kind == "demo":
                info = DEMOS[req.demo_id or ""]
                self._demo_zones = {k: [v] for k, v in info.zones.items()}  # one polygon each -> list of polygons
                if info.kind == "scripted":
                    gen = ScriptedDemo(info.id).frames_stream(req.loop)
                else:
                    source = FrameSource(info.path, pace_files=True, loop=req.loop)
            elif req.kind == "url":
                source = FrameSource(req.url, loop=False)
            elif req.kind == "webcam":
                source = FrameSource(str(req.index))
            else:
                gen = self._browser_frames()
            if gen is None:
                gen = self._capture(source)
            if not (req.kind == "demo" and DEMOS[req.demo_id or ""].kind == "scripted"):
                self._set_status("starting", "비전 모델을 불러오는 중… (처음에는 수십 초 걸릴 수 있습니다)")
                extractor = await loop.run_in_executor(self._pool, self._extractor, req.precision)
                proc = SpatialFeatureProcessor(self.st)
            self._set_status("running", "분석 중", source=req.kind, demo=req.demo_id, live=self.live,
                             zones_hint=self._demo_zones)
            idx, stamps, last_send, vision_ms = 0, deque(maxlen=30), 0.0, 0.0

            def process(frame: np.ndarray, ts: float, i: int) -> FrameFeatures:
                h, w = frame.shape[:2]
                return proc.process(extractor.infer(frame, i), (w, h), ts, i)  # type: ignore[union-attr]

            async for ts, img, feats, restarted in gen:
                if restarted:
                    pipe.reset()
                if feats is None:
                    t0 = time.perf_counter()
                    feats = await loop.run_in_executor(self._pool, process, img, ts, idx)
                    vision_ms = (time.perf_counter() - t0) * 1000
                self._wh = (img.shape[1], img.shape[0])
                await pipe.on_frame(feats, img)
                idx += 1
                stamps.append(time.monotonic())
                now = time.monotonic()
                if now - last_send >= 1 / SEND_FPS:
                    last_send = now
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
                    self._broadcast_frame(img, feats, pipe, fps, vision_ms)
            self._set_status("idle", "영상이 끝났습니다")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed")
            self._set_status("error", f"오류: {exc}")
        finally:
            if source:
                source.stop()
            try:
                await pipe.drain()
            except Exception:  # noqa: BLE001
                pass
            await predictor.aclose()
            self._ingest = None

    # ------------------------------------------------------------------ outputs

    def _broadcast_frame(self, img: np.ndarray, f: FrameFeatures, pipe: SafetyPipeline, fps: float, vision_ms: float) -> None:
        h, w = img.shape[:2]
        if w > SEND_MAX_W:
            img = cv2.resize(img, (SEND_MAX_W, round(h * SEND_MAX_W / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return
        self._last_jpeg = buf.tobytes()
        a = pipe.last_assessment
        workers = []
        for wk in f.workers:
            mine = [r for r in a.reasons if r.startswith(f"worker {wk.track_id}:")]
            urgent = (wk.nearest_equipment and wk.nearest_equipment.ttc_s is not None
                      and wk.nearest_equipment.ttc_s <= self.st.ttc_urgent_s) or (wk.torso_tilt_deg or 0) >= 20 and wk.on_ladder
            r = wk.nearest_equipment
            workers.append({
                "id": wk.track_id, "box": _norm(wk.box, *f.frame_wh), "zone": wk.zone,
                "flag": "danger" if urgent else "warn" if mine else "ok",
                "helmet": wk.helmet.status, "vest": wk.vest.status, "harness": wk.harness.status,
                "tilt": wk.torso_tilt_deg, "speed": wk.speed_mps, "stationary": wk.stationary_s, "facing": wk.facing,
                "aware": wk.aware_of_equipment, "ladder": wk.on_ladder, "climb": wk.climb_height_m,
                "eq": None if r is None else {"label": r.label, "dist": r.distance_m, "ttc": r.ttc_s, "closing": r.closing_speed_mps},
            })
        meta = {
            "fps": round(fps, 1), "vision_ms": round(vision_ms), "busy": pipe.busy, "workers": workers,
            "equipment": [{"id": e.eq_id, "label": e.label, "motion": e.motion, "box": _norm(e.box, *f.frame_wh)} for e in f.equipment],
            "zones": {"danger": self.st.danger_zones or self._demo_zones.get("danger", []),
                      "rest": self.st.rest_zones or self._demo_zones.get("rest", [])},
            "risk": {"reasons": a.reasons, "urgent": a.urgent, "fallback": a.fallback_worker is not None},
        }
        head = json.dumps(meta, ensure_ascii=False).encode()
        payload = struct.pack(">I", len(head)) + head + self._last_jpeg
        for c in self.clients:
            c.send_frame(payload)

    def _on_result(self, d: JevDecision, seq: SequencePayload, acts: list[Action], a: RiskAssessment, roi: RoiImage | None) -> None:
        ts, (w, h) = time.time(), self._wh
        snap = None
        data = roi.jpeg if roi else (self._last_jpeg or None)
        if data and (roi or any(x.kind == "dashboard_review" for x in acts)):
            name = f"{'roi' if roi else 'frame'}_{int(ts * 1000)}.jpg"
            (self.snap_dir / name).write_bytes(data)
            snap = f"/snapshots/{name}"
        self._emit({
            "kind": "decision", "ts": ts, "mode": d.mode, "p": round(d.p_hazard, 3), "risk": round(d.risk_score, 2),
            "plan": d.action_plan, "plan_ko": PLAN_KO.get(d.action_plan, d.action_plan), "latency_ms": round(d.latency_ms),
            "tokens_in": d.usage.input_tokens, "tokens_out": d.usage.output_tokens, "model": d.model, "live": self.live,
            "reasons": a.reasons, "snapshot": snap,
            "narratives": [{"id": x.track_id, "text": x.narrative} for x in seq.workers],
            "roi_box": None if d.roi_box is None else _norm(d.roi_box, w, h), "roi_conf": d.roi_conf,
        })
        for act in acts:
            if act.kind != "monitor":
                self._emit({"kind": "action", "ts": ts, "action": act.kind, "tier": act.tier, "label": ACTION_KO[act.kind],
                            "reason": act.reason, "p": act.p_hazard, "risk": act.risk_score, "mode": d.mode, "snapshot": snap})


def probe_cameras(max_index: int = 4) -> list[dict]:
    """Which server-side camera indices open? (macOS needs camera permission for the process running the server.)"""
    out = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        item = {"index": i, "ok": ok}
        if ok:
            r, fr = cap.read()
            item["ok"] = bool(r)
            if r:
                item["size"] = f"{fr.shape[1]}x{fr.shape[0]}"
        cap.release()
        out.append(item)
    return out
