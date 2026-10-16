"""Spatial/geometry layer: raw detections -> per-worker safety features.

- matches small PPE (helmet / vest / harness) to the person that wears it (head and torso regions of the person box)
- keeps borderline detections as "uncertain (conf: x)" instead of dropping them
- derives posture, gait, gaze, stationary time, ladder height, worker<->equipment distance, closing speed, TTC
- distances are monocular estimates: metres come from the person's pixel height (person_height_m)
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .config import Polygon, Settings
from .schemas import (BBox, Detection, EquipmentFeatures, EquipmentRelation, FrameFeatures, Keypoint, PPEItem,
                      PPEState, RiskAssessment, WorkerFeatures)

EQUIPMENT = {"excavator", "forklift", "crane hook"}
STRUCTURES = {"ladder", "scaffolding"}
_NOSE, _L_EYE, _R_EYE, _L_SH, _R_SH, _L_WR, _R_WR, _L_HIP, _R_HIP = 0, 1, 2, 5, 6, 9, 10, 11, 12


class _CentroidTracker:
    """Id assignment for detections that arrive without a tracker id (equipment, or persons when track A is off)."""

    def __init__(self, base_id: int, max_frac: float = 0.12, max_age_s: float = 1.0):
        self.next, self.max_frac, self.max_age = base_id, max_frac, max_age_s
        self.tracks: dict[int, tuple[str, float, float, float]] = {}

    def assign(self, dets: list[Detection], t: float, diag: float) -> list[int]:
        self.tracks = {k: v for k, v in self.tracks.items() if t - v[3] <= self.max_age}
        ids: list[int] = [0] * len(dets)
        used: set[int] = set()
        for i in sorted(range(len(dets)), key=lambda i: -dets[i].conf):
            cx, cy = dets[i].box.center
            best, bd = None, self.max_frac * diag
            for tid, (lab, tx, ty, _) in self.tracks.items():
                if tid in used or lab != dets[i].label:
                    continue
                d = math.hypot(cx - tx, cy - ty)
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best, self.next = self.next, self.next + 1
            used.add(best)
            ids[i] = best
            self.tracks[best] = (dets[i].label, cx, cy, t)
        return ids


def _poly_contains(poly: Polygon, x: float, y: float, w: int, h: int) -> bool:
    pts = (np.array(poly, dtype=np.float32) * np.array([w, h], dtype=np.float32)).astype(np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def pose_features(kp: list[Keypoint] | None, ref_x: float) -> tuple[float | None, str, float | None]:
    """-> (torso tilt deg from vertical, facing, shoulder-mid x). Needs keypoint confidence >= 0.3 (face >= 0.5)."""
    if not kp or len(kp) < 13:
        return None, "unknown", None
    ok = lambda i, c=0.3: kp[i].conf >= c  # noqa: E731
    sh = [kp[i] for i in (_L_SH, _R_SH) if ok(i)]
    hip = [kp[i] for i in (_L_HIP, _R_HIP) if ok(i)]
    tilt = sx = None
    if sh:
        sx = sum(k.x for k in sh) / len(sh)
    if sh and hip:
        sy = sum(k.y for k in sh) / len(sh)
        hx, hy = sum(k.x for k in hip) / len(hip), sum(k.y for k in hip) / len(hip)
        dy = hy - sy
        tilt = 90.0 if dy <= 1 else math.degrees(math.atan2(abs(hx - sx), dy))
    nose, eyes = ok(_NOSE, 0.5), [ok(_L_EYE, 0.5), ok(_R_EYE, 0.5)]
    if nose and all(eyes):
        facing = "toward_camera"
    elif nose and any(eyes):
        facing = "left" if kp[_NOSE].x < (sx if sx is not None else ref_x) else "right"
    elif not nose and not any(eyes) and sh:
        facing = "away_from_camera"
    else:
        facing = "unknown"
    return tilt, facing, sx


def awareness(facing: str, worker_foot: tuple[float, float], eq_box: BBox, frame_h: int) -> bool | None:
    """Is the worker plausibly looking towards the equipment? Heuristic from 2D gaze, None when undecidable."""
    wx, wy = worker_foot
    if facing in ("left", "right"):
        dx = eq_box.center[0] - wx
        if abs(dx) < 0.03 * frame_h:
            return None
        return dx < 0 if facing == "left" else dx > 0
    depth = eq_box.y2 - wy  # >0: equipment is nearer the camera than the worker
    if abs(depth) < 0.03 * frame_h:
        return None
    if facing == "toward_camera":
        return depth > 0
    if facing == "away_from_camera":
        return depth < 0
    return None


class SpatialFeatureProcessor:
    def __init__(self, st: Settings):
        self.st = st
        self._ptrack = _CentroidTracker(base_id=1000)
        self._etrack = _CentroidTracker(base_id=1)
        self._wpos: dict[int, deque] = {}
        self._wspeed: dict[int, deque] = {}
        self._anchor: dict[int, tuple[float, float, float]] = {}
        self._epos: dict[int, deque] = {}
        self._dist: dict[tuple[int, int], deque] = {}
        self._seen: dict[object, float] = {}

    # ------------------------------------------------------------------ public

    def process(self, dets: list[Detection], frame_wh: tuple[int, int], t: float, idx: int) -> FrameFeatures:
        w, h = frame_wh
        diag = math.hypot(w, h)
        persons = [d for d in dets if d.label == "person"]
        eqs = [d for d in dets if d.label in EQUIPMENT]
        structs = [d for d in dets if d.label in STRUCTURES]
        gear = [d for d in dets if d.label != "person" and d.label not in EQUIPMENT | STRUCTURES]

        untracked = [d for d in persons if d.track_id is None]
        for d, tid in zip(untracked, self._ptrack.assign(untracked, t, diag)):
            d.track_id = tid
        eq_ids = self._etrack.assign(eqs, t, diag)
        eq_items = list(zip(eq_ids, eqs))
        for eid, e in eq_items:
            self._push(self._epos, eid, (t, *e.box.foot), t)

        workers = [self._worker(d, t, frame_wh, eq_items, structs, gear) for d in persons]
        equipment = [self._equipment(eid, e, workers) for eid, e in eq_items]
        if idx % 150 == 0:
            self._gc(t)
        return FrameFeatures(t=t, frame_idx=idx, frame_wh=frame_wh, workers=sorted(workers, key=lambda x: x.track_id),
                             equipment=equipment)

    # ------------------------------------------------------------------ workers

    def _worker(self, d: Detection, t: float, frame_wh: tuple[int, int], eq_items, structs, gear) -> WorkerFeatures:
        w, h = frame_wh
        tid = int(d.track_id or 0)
        gx, gy = d.box.foot
        mpp = self.st.person_height_m / max(d.box.h, 20.0)  # metres per pixel at this person's depth
        self._push(self._wpos, tid, (t, gx, gy, mpp), t)

        speed = self._speed(self._wpos[tid])
        self._push(self._wspeed, tid, (t, speed), t)
        sp = [v for ts, v in self._wspeed[tid] if t - ts <= 2.0]
        gait = float(np.std(sp) / (np.mean(sp) + 0.3)) if len(sp) >= 6 else 0.0

        anchor = self._anchor.get(tid)
        if anchor is None or math.hypot(gx - anchor[0], gy - anchor[1]) * mpp > 0.35:
            anchor = self._anchor[tid] = (gx, gy, t)
        stationary = t - anchor[2]

        zone = "general"
        if any(_poly_contains(p, gx, gy, w, h) for p in self.st.rest_zones):
            zone = "rest"
        if any(_poly_contains(p, gx, gy, w, h) for p in self.st.danger_zones):
            zone = "danger"

        tilt, facing, sx = pose_features(d.keypoints, d.box.center[0])
        helmet, vest, harness = self._ppe(d, gear, structs)

        on_ladder, climb, outside = False, None, None
        for s in structs:
            if s.label != "ladder":
                continue
            if d.box.inter(s.box) / max(d.box.area, 1.0) > 0.25 or s.box.rows(0, 1, grow_x=0.3).contains(gx, gy):
                on_ladder = True
                climb = max(0.0, s.box.y2 - gy) * mpp
                if sx is not None:
                    outside = bool(sx < s.box.x1 or sx > s.box.x2)
                break

        rel = self._nearest_equipment(tid, gx, gy, mpp, t, eq_items)
        aware = awareness(facing, (gx, gy), rel.box, h) if rel and rel.box else None
        return WorkerFeatures(track_id=tid, box=d.box, zone=zone, speed_mps=round(speed, 2),  # type: ignore[arg-type]
                              gait_variability=round(gait, 2), stationary_s=round(stationary, 1),
                              torso_tilt_deg=None if tilt is None else round(tilt, 1), facing=facing,  # type: ignore[arg-type]
                              aware_of_equipment=aware, helmet=helmet, vest=vest, harness=harness, on_ladder=on_ladder,
                              climb_height_m=None if climb is None else round(climb, 1), torso_outside_ladder=outside,
                              nearest_equipment=rel)

    def _speed(self, hist: deque) -> float:
        t1, x1, y1, m1 = hist[-1]
        for t0, x0, y0, m0 in hist:
            if t1 - t0 <= 1.0:
                if t1 - t0 < 0.15:
                    return 0.0
                return min(8.0, math.hypot(x1 - x0, y1 - y0) * (m0 + m1) / 2 / (t1 - t0))
        return 0.0

    # ------------------------------------------------------------------ PPE matching (spatial hierarchy)

    def _grade(self, conf: float, good: PPEState) -> PPEItem:
        return PPEItem(state=good if conf >= self.st.conf_confident else PPEState.UNCERTAIN, conf=round(conf, 2))

    def _ppe(self, p: Detection, gear: list[Detection], structs: list[Detection]):
        head = p.box.rows(0.0, 0.30, grow_x=0.15, grow_y=0.10)  # top 30%: helmet
        torso = p.box.rows(0.25, 0.75, grow_x=0.10)  # middle: vest, harness
        body = p.box.rows(0.0, 1.0, grow_x=0.15, grow_y=0.05)

        def inside(det: Detection, region: BBox) -> bool:
            return region.contains(*det.box.center) or region.iou(det.box) > 0.15

        def best(label: str, region: BBox) -> Detection | None:
            c = [g for g in gear if g.label == label and inside(g, region)]
            return max(c, key=lambda g: g.conf) if c else None

        pos, neg = best("hard hat", head), best("no hard hat", head)
        wrists = [p.keypoints[i] for i in (_L_WR, _R_WR) if p.keypoints and len(p.keypoints) > i and p.keypoints[i].conf >= 0.3]
        held = [g for g in gear if g.label == "hard hat" and not inside(g, head)
                and any(math.hypot(g.box.center[0] - k.x, g.box.center[1] - k.y) < 0.25 * p.box.h for k in wrists)]
        if neg and (not pos or neg.conf > pos.conf):
            helmet = self._grade(neg.conf, PPEState.ABSENT)
        elif pos:
            helmet = self._grade(pos.conf, PPEState.WORN)
        elif held:
            helmet = PPEItem(state=PPEState.HELD, conf=round(max(g.conf for g in held), 2))
        else:
            helmet = PPEItem()

        v_pos, v_neg = best("safety vest", torso), best("no safety vest", torso)
        if v_neg and (not v_pos or v_neg.conf > v_pos.conf):
            vest = self._grade(v_neg.conf, PPEState.ABSENT)
        elif v_pos:
            vest = self._grade(v_pos.conf, PPEState.WORN)
        else:
            vest = PPEItem()

        reach = p.box.rows(-0.7, 1.0, grow_x=0.3)  # a hook sits on the structure, often above the worker's head
        hook, lan = best("safety hook", reach), best("harness lanyard", body)
        attached = hook is not None and any(s.box.inter(hook.box) > 0 for s in structs)
        if hook and attached:
            harness = self._grade(hook.conf, PPEState.HOOKED)
        elif hook or lan:
            harness = self._grade(max(x.conf for x in (hook, lan) if x), PPEState.UNHOOKED)  # lanyard seen, no attachment seen
        else:
            harness = PPEItem()
        return helmet, vest, harness

    # ------------------------------------------------------------------ equipment relations

    def _nearest_equipment(self, tid: int, gx: float, gy: float, mpp: float, t: float, eq_items) -> EquipmentRelation | None:
        best: EquipmentRelation | None = None
        for eid, e in eq_items:
            b = e.box
            dx = max(b.x1 - gx, 0.0, gx - b.x2)
            dy = max((b.y2 - 0.35 * b.h) - gy, 0.0, gy - b.y2)  # compare with the equipment's ground band
            d = math.hypot(dx, dy) * mpp
            hist = self._push(self._dist, (tid, eid), (t, d), t)
            closing = 0.0
            for t0, d0 in hist:
                if t - t0 <= 1.0:
                    if t - t0 >= 0.25:
                        closing = (d0 - d) / (t - t0)
                    break
            ttc = round(min(30.0, d / closing), 1) if closing > 0.3 else None
            rel = EquipmentRelation(eq_id=eid, label=e.label, distance_m=round(d, 1), closing_speed_mps=round(closing, 2),
                                    ttc_s=ttc, box=b)
            if best is None or rel.distance_m < best.distance_m:
                best = rel
        return best

    def _equipment(self, eid: int, e: Detection, workers: list[WorkerFeatures]) -> EquipmentFeatures:
        hist = self._epos[eid]
        mpp = self._wpos[workers[0].track_id][-1][3] if workers else self.st.person_height_m / (0.25 * 720)
        t1, x1, y1 = hist[-1]
        speed = 0.0
        for t0, x0, y0 in hist:
            if t1 - t0 <= 1.0 and t1 - t0 >= 0.15:
                speed = math.hypot(x1 - x0, y1 - y0) * mpp / (t1 - t0)
                break
        closing = max((w.nearest_equipment.closing_speed_mps for w in workers
                       if w.nearest_equipment and w.nearest_equipment.eq_id == eid), default=0.0)
        if speed < 0.25:
            motion = "stationary"
        elif closing > 0.2:
            motion = "approaching"
        elif closing < -0.2:
            motion = "receding"
        else:
            motion = "moving"
        return EquipmentFeatures(eq_id=eid, label=e.label, box=e.box, speed_mps=round(min(speed, 8.0), 2), motion=motion)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ bookkeeping

    def _push(self, store: dict, key, item, t: float) -> deque:
        q = store.setdefault(key, deque(maxlen=90))
        q.append(item)
        self._seen[(id(store), key)] = t
        return q

    def _gc(self, now: float) -> None:
        stale = {k for k, v in self._seen.items() if now - v > 3.0}
        for store in (self._wpos, self._wspeed, self._epos, self._dist):
            for k in [k for k in store if (id(store), k) in stale]:
                del store[k]
        for k in stale:
            self._seen.pop(k, None)


# ----------------------------------------------------------------------------- risk assessment


def assess(f: FrameFeatures, st: Settings) -> RiskAssessment:
    """Cheap per-frame triage: is there a risky context, is it urgent, and is the vision evidence too weak to trust?"""
    out = RiskAssessment()
    for w in f.workers:
        why: list[str] = []
        eq = w.nearest_equipment
        if eq and eq.ttc_s is not None and eq.ttc_s <= st.ttc_warn_s:
            why.append(f"worker {w.track_id}: {eq.label} TTC {eq.ttc_s:.1f}s")
            out.urgent |= eq.ttc_s <= st.ttc_urgent_s
        elif eq and eq.distance_m <= st.near_distance_m and eq.closing_speed_mps > 0.2:
            why.append(f"worker {w.track_id}: {eq.label} {eq.distance_m:.1f}m and closing")
        if w.zone == "danger":
            why.append(f"worker {w.track_id}: in danger zone")
        bad_hook = w.harness.state in (PPEState.UNHOOKED, PPEState.NOT_DETECTED, PPEState.UNCERTAIN)
        if w.on_ladder and (w.climb_height_m or 0) >= 2.0 and bad_hook:
            why.append(f"worker {w.track_id}: {w.climb_height_m:.1f}m up a ladder, harness not confirmed hooked")
        tilt = w.torso_tilt_deg or 0.0
        if (w.on_ladder and tilt >= 20) or tilt >= 60:
            why.append(f"worker {w.track_id}: torso tilt {tilt:.0f} deg")
            out.urgent = True
        out.reasons += why
        weak = any(i.state is PPEState.UNCERTAIN and (i.conf or 0) < st.fallback_conf for i in (w.helmet, w.vest, w.harness))
        if why and weak and out.fallback_worker is None:
            out.fallback_worker = w.track_id
    return out
