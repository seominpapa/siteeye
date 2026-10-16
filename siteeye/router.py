"""Turns a Jev decision into actions. Tiers follow the spec:

  P(yes) >= 0.80 or risk_score >= 4.0   -> equipment interlock + site speaker (+ worker smartwatch)
  0.50 <= P(yes) < 0.80                 -> safety manager dashboard with snapshot + sequence log (human in the loop)
  P(yes) < 0.50                         -> keep monitoring (false-alarm suppression)

Sinks are pluggable. ConsoleSink only logs; WebhookSink posts JSON to URLs you configure. Wiring a real
equipment interlock is a site-engineering job (fail-safe design, independent safety PLC): treat this pipeline
as an advisory layer, never as the only protection.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .jev_client import ACTIONS, RoiImage
from .schemas import JevDecision, SequencePayload

log = logging.getLogger("siteeye.router")
Kind = Literal["equipment_estop", "speaker_alert", "smartwatch_vibration", "dashboard_review", "monitor"]


class Action(BaseModel):
    kind: Kind
    tier: Literal["immediate", "review", "monitor"]
    reason: str
    p_hazard: float
    risk_score: float
    jev_action_plan: str
    payload: dict = Field(default_factory=dict)


class ActionSink(Protocol):
    async def emit(self, action: Action) -> None: ...


class ConsoleSink:
    async def emit(self, action: Action) -> None:
        if action.kind != "monitor":
            print(f"  [ACTION:{action.tier.upper()}] {action.kind}  ({action.reason})")


class WebhookSink:
    """POSTs actions to per-kind URLs, e.g. {"dashboard_review": "https://dash.example/api/alerts"}."""

    def __init__(self, urls: dict[str, str], client: httpx.AsyncClient | None = None):
        self.urls, self.client = urls, client or httpx.AsyncClient(timeout=3.0)

    async def emit(self, action: Action) -> None:
        url = self.urls.get(action.kind)
        if not url:
            return
        try:
            await self.client.post(url, json=action.model_dump(mode="json"))
        except httpx.HTTPError as exc:
            log.error("webhook %s failed: %s", action.kind, exc)


class SafetyActionRouter:
    def __init__(self, st: Settings, sinks: list[ActionSink] | None = None):
        self.st = st
        self.sinks = [ConsoleSink()] if sinks is None else sinks
        self._last: dict[str, float] = {}

    def route(self, d: JevDecision, seq: SequencePayload, roi: RoiImage | None = None) -> list[Action]:
        st = self.st
        base = dict(p_hazard=round(d.p_hazard, 3), risk_score=round(d.risk_score, 2), jev_action_plan=d.action_plan)
        why = f"P(yes)={d.p_hazard:.2f}, risk={d.risk_score:.1f}/5, jev plan={ACTIONS.get(d.action_plan, d.action_plan)}"
        if d.p_hazard >= st.p_immediate or d.risk_score >= st.risk_immediate:
            kinds = ["speaker_alert", "smartwatch_vibration"]
            if seq.equipment:  # nothing to stop for a fall-from-height with no machine in the scene
                kinds.insert(0, "equipment_estop")
            acts = [Action(kind=k, tier="immediate", reason=why, **base) for k in kinds]
        elif d.p_hazard >= st.p_review:
            snap = self._save_snapshot(roi) if roi else None
            acts = [Action(kind="dashboard_review", tier="review", reason=why, **base,
                           payload={"sequence": seq.model_dump(mode="json"), "snapshot": snap,
                                    "roi_box": d.roi_box.model_dump() if d.roi_box else None})]
        else:
            return [Action(kind="monitor", tier="monitor", reason=why, **base)]
        return [a for a in acts if self._allow(a)]

    def _allow(self, a: Action) -> bool:
        """Interlock is idempotent and always passes; operator-facing alerts are rate limited."""
        if a.kind == "equipment_estop":
            return True
        gap = {"speaker_alert": 10.0, "smartwatch_vibration": 5.0, "dashboard_review": 20.0}[a.kind]
        now = time.monotonic()
        if now - self._last.get(a.kind, -1e9) < gap:
            return False
        self._last[a.kind] = now
        return True

    def _save_snapshot(self, roi: RoiImage) -> str:
        d = self.st.out_dir / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        p: Path = d / f"review_{int(time.time() * 1000)}.jpg"
        p.write_bytes(roi.jpeg)
        return str(p)

    async def dispatch(self, actions: list[Action]) -> None:
        for a in actions:
            for s in self.sinks:
                await s.emit(a)
