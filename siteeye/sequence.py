"""Sliding-window sequence buffer: the last N seconds as 1-second snapshots -> causal-narrative JSON for Jev.

The point is context, not coordinates: "stopped, then took the helmet off in a rest zone" and
"stopped, then helmet unrecognised while an excavator closes in" differ only in the order of events.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import Settings
from .schemas import (EquipmentFeatures, EquipmentSummary, FrameFeatures, Moment, PPEState, SequencePayload,
                       WorkerFeatures, WorkerTimeline)

LABEL_KO = {"excavator": "굴착기", "forklift": "지게차", "crane hook": "크레인 훅"}
_NOT_WORN = (PPEState.ABSENT, PPEState.HELD, PPEState.UNCERTAIN)


@dataclass
class _Snap:
    t: float
    workers: dict[int, WorkerFeatures] = field(default_factory=dict)
    # worst case within the second, so a brief spike is not lost by keeping only the last frame
    min_ttc: dict[int, float] = field(default_factory=dict)
    max_tilt: dict[int, float] = field(default_factory=dict)
    equipment: list[EquipmentFeatures] = field(default_factory=list)


class SequenceBuffer:
    def __init__(self, st: Settings):
        self.st = st
        self._snaps: deque[_Snap] = deque(maxlen=st.window_s)  # finalized seconds
        self._cur: _Snap | None = None
        self._bucket: int | None = None

    def __len__(self) -> int:
        return len(self._snaps) + (self._cur is not None)

    def push(self, f: FrameFeatures) -> None:
        bucket = int(f.t // self.st.snapshot_interval_s)
        if self._bucket is not None and bucket != self._bucket and self._cur is not None:
            self._snaps.append(self._cur)
            self._cur = None
        if self._cur is None:
            self._cur = _Snap(t=f.t)
        self._bucket, c = bucket, self._cur
        c.t, c.equipment = f.t, f.equipment
        for w in f.workers:
            c.workers[w.track_id] = w
            ttc = w.nearest_equipment.ttc_s if w.nearest_equipment else None
            if ttc is not None:
                c.min_ttc[w.track_id] = min(ttc, c.min_ttc.get(w.track_id, ttc))
            if w.torso_tilt_deg is not None:
                c.max_tilt[w.track_id] = max(w.torso_tilt_deg, c.max_tilt.get(w.track_id, 0.0))

    # ------------------------------------------------------------------ payload

    def build(self) -> SequencePayload:
        snaps = list(self._snaps) + ([self._cur] if self._cur else [])
        snaps = snaps[-self.st.window_s:]
        t_now = snaps[-1].t
        labels = [f"T-{round((t_now - s.t) / self.st.snapshot_interval_s)}s" for s in snaps]
        ids = sorted({i for s in snaps for i in s.workers})
        timelines = [self._timeline(i, snaps, labels) for i in ids if i in snaps[-1].workers]
        eq = [EquipmentSummary(eq_id=e.eq_id, label=e.label, motion=e.motion, speed_mps=e.speed_mps)
              for e in snaps[-1].equipment]
        return SequencePayload(camera=self.st.camera, window=f"{labels[0]}..{labels[-1]}", site_notes=self.st.site_notes,
                               workers=timelines, equipment=eq)

    def _timeline(self, tid: int, snaps: list[_Snap], labels: list[str]) -> WorkerTimeline:
        moments: list[Moment] = []
        prev: WorkerFeatures | None = None
        for s, lab in zip(snaps, labels):
            w = s.workers.get(tid)
            if w is None:
                continue
            tilt = s.max_tilt.get(tid, w.torso_tilt_deg)
            rel = w.nearest_equipment
            if rel is not None and tid in s.min_ttc:
                rel = rel.model_copy(update={"ttc_s": s.min_ttc[tid]})
            moments.append(Moment(
                t=lab, zone=w.zone, speed_mps=w.speed_mps, stationary_s=w.stationary_s,
                tilt_deg=None if tilt is None else round(tilt, 1), facing=w.facing,
                aware_of_equipment=w.aware_of_equipment, helmet_status=w.helmet.status, vest_status=w.vest.status,
                harness_status=w.harness.status, on_ladder=w.on_ladder, climb_height_m=w.climb_height_m,
                torso_outside_ladder=w.torso_outside_ladder, nearest_equipment=rel,
                events=self._events(prev, w, rel)))
            prev = w
        return WorkerTimeline(track_id=tid, narrative=self._narrative(moments), moments=moments)

    # ------------------------------------------------------------------ event extraction

    @staticmethod
    def _events(prev: WorkerFeatures | None, w: WorkerFeatures, rel) -> list[str]:
        ev: list[str] = []
        if w.zone != (prev.zone if prev else "general"):
            ev.append({"rest": "휴게구역 진입", "danger": "위험구역 진입", "general": "구역 이탈"}[w.zone])
        if w.stationary_s >= 1.5 and (prev is None or prev.stationary_s < 1.5):
            ev.append("이동 정지")
        was_worn = prev is None or prev.helmet.state not in _NOT_WORN
        if w.helmet.state is PPEState.HELD and (prev is None or prev.helmet.state is not PPEState.HELD):
            ev.append("안전모 탈착 및 손에 파지")
        elif w.helmet.state is PPEState.ABSENT and was_worn:
            ev.append("안전모 미착용 상태")
        elif w.helmet.state is PPEState.UNCERTAIN and (prev is None or prev.helmet.state is not PPEState.UNCERTAIN):
            ev.append(f"안전모 {w.helmet.status}")
        if w.on_ladder:
            if prev is None or not prev.on_ladder:
                ev.append(f"사다리 {w.climb_height_m or 0:.1f}m 등반")
            if w.harness.state is not PPEState.HOOKED and (prev is None or prev.harness.state is not w.harness.state):
                ev.append(f"안전고리 미결속 상태 ({w.harness.status})")
        if w.torso_tilt_deg is not None and w.torso_tilt_deg >= 15:
            ev.append(f"상체 기울기 {w.torso_tilt_deg:.0f}도" + (" (사다리 외곽 이탈)" if w.torso_outside_ladder else ""))
        if rel is not None and rel.distance_m <= 8.0:
            name = LABEL_KO.get(rel.label, rel.label)
            msg = f"{name} 거리 {rel.distance_m:.1f}m"
            if rel.closing_speed_mps > 0.2:
                msg += " 접근"
            if rel.ttc_s is not None and rel.ttc_s <= 5.0:
                msg += f", 예상 충돌 {rel.ttc_s:.1f}초 전"
            ev.append(msg)
            if w.aware_of_equipment is False:
                ev.append("작업자 시선이 장비 반대 방향 (장비 인지 못함)")
        return ev

    @staticmethod
    def _current_state(m: Moment) -> list[str]:
        """What is true right now, used for the final step when no new event happened in that second."""
        s: list[str] = []
        if m.zone != "general":
            s.append({"rest": "휴게구역 내", "danger": "위험구역 내"}[m.zone])
        if m.stationary_s >= 1.5:
            s.append(f"{m.stationary_s:.0f}초째 정지")
        helmet = {"absent": "안전모 미착용", "held_in_hand": "안전모 손에 파지"}.get(m.helmet_status)
        if helmet or m.helmet_status.startswith("uncertain"):
            s.append(helmet or f"안전모 {m.helmet_status}")
        if m.on_ladder and m.harness_status != "hooked":
            s.append("안전고리 " + {"unhooked": "미결속", "not_detected": "미확인"}.get(m.harness_status, m.harness_status))
        return s

    @classmethod
    def _narrative(cls, moments: list[Moment]) -> str:
        parts = []
        for i, m in enumerate(moments):
            last = i == len(moments) - 1
            ev = m.events or (cls._current_state(m) if last else [])
            if ev or last:
                parts.append(f"[{m.t}: {', '.join(ev) or '특이사항 없음'}]")
        return " -> ".join(parts)
