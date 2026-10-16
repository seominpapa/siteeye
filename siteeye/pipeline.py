"""Async orchestration: per-frame features -> sequence buffer -> throttled Jev calls -> routed actions.

Vision runs at the camera frame rate (30 FPS). Jev is called at most every `jev_interval_s`, or sooner when
`assess()` flags an urgent situation (never faster than `jev_urgent_min_gap_s`), with one call in flight at a
time, so a slow API call can never stall or queue up behind the video loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import numpy as np

from .config import Settings
from .jev_client import JevSafetyPredictor, RoiImage, prepare_roi
from .router import Action, SafetyActionRouter
from .schemas import BBox, FrameFeatures, JevDecision, RiskAssessment, SequencePayload
from .sequence import SequenceBuffer
from .spatial import assess

log = logging.getLogger("siteeye.pipeline")


def focus_box(f: FrameFeatures, worker_id: int) -> BBox:
    """ROI = the worker plus the equipment closing in on them, so Jev sees both."""
    w = next(w for w in f.workers if w.track_id == worker_id)
    box = w.box
    if w.nearest_equipment and w.nearest_equipment.box is not None:
        box = box.union(w.nearest_equipment.box)
    return box


class SafetyPipeline:
    def __init__(self, st: Settings, predictor: JevSafetyPredictor, router: SafetyActionRouter,
                 verbose: bool = True, log_path: str | None = None):
        self.st, self.predictor, self.router, self.verbose = st, predictor, router, verbose
        self.buffer = SequenceBuffer(st)
        self.last: JevDecision | None = None
        self.last_actions: list[Action] = []
        self.last_assessment = RiskAssessment()
        # (decision, sequence, actions, assessment, roi) after every Jev call; the web UI hooks in here
        self.on_result: Callable[[JevDecision, SequencePayload, list[Action], RiskAssessment, RoiImage | None], None] | None = None
        self.gate: Callable[[], bool] | None = None  # return False to skip a Jev call (budget exhausted)
        self._task: asyncio.Task | None = None
        self._last_call = -1e9
        self._log = open(log_path, "a", buffering=1) if log_path else None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def reset(self) -> None:
        """Forget the sequence window (new source, or a looping demo restarting)."""
        self.buffer = SequenceBuffer(self.st)
        self._last_call = -1e9

    # ------------------------------------------------------------------ streaming path

    async def on_frame(self, f: FrameFeatures, frame: np.ndarray) -> None:
        a = self.last_assessment = assess(f, self.st)
        self.buffer.push(f)
        gap = f.t - self._last_call
        due = gap >= self.st.jev_interval_s or (a.urgent and gap >= self.st.jev_urgent_min_gap_s)
        if not due or not f.workers or (self._task and not self._task.done()):
            return
        if self.gate and not self.gate():
            return
        self._last_call = f.t
        roi = self._roi(f, frame, a)  # crop now: the frame buffer is reused by the capture thread
        self._task = asyncio.create_task(self._safe_decide(self.buffer.build(), roi, a))

    async def drain(self) -> None:
        if self._task:
            await self._task

    async def _safe_decide(self, seq: SequencePayload, roi: RoiImage | None, a: RiskAssessment) -> None:
        try:
            await self.decide(seq, roi, a)
        except Exception as exc:  # noqa: BLE001 - network/API trouble must not stop detection
            log.error("Jev call failed: %s", exc)

    # ------------------------------------------------------------------ one decision (also used by --mock)

    def _roi(self, f: FrameFeatures, frame: np.ndarray, a: RiskAssessment) -> RoiImage | None:
        if a.fallback_worker is None:
            return None
        return prepare_roi(frame, focus_box(f, a.fallback_worker))

    async def decide(self, seq: SequencePayload, roi: RoiImage | None, a: RiskAssessment) -> JevDecision:
        d = await self.predictor.predict(seq, roi)
        acts = self.router.route(d, seq, roi)
        self.last, self.last_actions = d, acts
        if self.verbose:
            tag = "IMAGE-FALLBACK" if d.mode == "image_fallback" else "text"
            print(f"[jev:{tag}] P(hazard)={d.p_hazard:.2f} risk={d.risk_score:.1f}/5 plan={d.action_plan} "
                  f"{d.latency_ms:.0f}ms tokens={d.usage.input_tokens}+{d.usage.output_tokens}"
                  + (f"  why: {'; '.join(a.reasons)}" if a.reasons else ""))
        if self.on_result:
            self.on_result(d, seq, acts, a, roi)
        if self._log:
            self._log.write(json.dumps({"decision": d.model_dump(mode="json"), "actions": [x.kind for x in acts],
                                        "reasons": a.reasons}, ensure_ascii=False) + "\n")
        await self.router.dispatch(acts)
        return d
