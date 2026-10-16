"""--mock mode: four scripted 5-second scenarios, no camera or GPU needed.

Each scenario is a list of FrameFeatures (one per second, T-4s..T-0s) plus a synthetic frame used for the ROI crop.
Without TYPESAFE_API_KEY (or with --offline) the calls go to a local stand-in responder over httpx.MockTransport.
That responder is a few if-statements, NOT Jev: it only exercises request building, headers, parsing and routing.
Run with a real key to see Jev's own judgement on the same payloads.
"""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from .config import Settings
from .jev_client import ACTIONS, IMAGES_HEADER, JevSafetyPredictor, prepare_roi
from .pipeline import SafetyPipeline, focus_box
from .router import SafetyActionRouter
from .schemas import (BBox, EquipmentFeatures, EquipmentRelation, FrameFeatures, PPEItem, PPEState, WorkerFeatures)
from .sequence import SequenceBuffer
from .spatial import assess

W, H = 1280, 720
T0 = 1000.0


@dataclass
class Scenario:
    id: int
    title: str
    expect: str
    frames: list[FrameFeatures]
    image: np.ndarray


def _ppe(state: PPEState, conf: float | None = None) -> PPEItem:
    return PPEItem(state=state, conf=conf)


def _frame(i: int, workers: list[WorkerFeatures], equipment: list[EquipmentFeatures] | None = None) -> FrameFeatures:
    return FrameFeatures(t=T0 + i, frame_idx=i * 30, frame_wh=(W, H), workers=workers, equipment=equipment or [])


def _rel(eid: int, box: BBox, d: float, closing: float, label: str = "excavator") -> EquipmentRelation:
    ttc = round(d / closing, 1) if closing > 0.3 else None
    return EquipmentRelation(eq_id=eid, label=label, distance_m=d, closing_speed_mps=closing, ttc_s=ttc, box=box)


def _canvas(backlit: bool = False) -> np.ndarray:
    img = np.full((H, W, 3), (150, 140, 130), np.uint8)
    img[: H // 2] = np.linspace((235, 215, 190) if backlit else (205, 185, 160), (150, 140, 130), H // 2)[:, None, :].astype(np.uint8)
    cv2.rectangle(img, (0, 560), (W, H), (95, 100, 105), -1)
    cv2.putText(img, "SYNTHETIC MOCK FRAME", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def _person(img: np.ndarray, box: BBox, dark: bool) -> None:
    col = (25, 25, 30) if dark else (60, 110, 190)
    x, y, w, h = int(box.x1), int(box.y1), int(box.w), int(box.h)
    cv2.rectangle(img, (x + w // 6, y + h // 4), (x + 5 * w // 6, y + h), col, -1)
    cv2.circle(img, (x + w // 2, y + h // 8), h // 9, (20, 20, 20) if dark else (200, 170, 140), -1)


def _machine(img: np.ndarray, box: BBox, dark: bool) -> None:
    col = (30, 30, 35) if dark else (40, 170, 230)
    cv2.rectangle(img, (int(box.x1), int(box.y1 + box.h * 0.35)), (int(box.x2), int(box.y2)), col, -1)
    cv2.rectangle(img, (int(box.x1), int(box.y1 + box.h * 0.1)), (int(box.x1 + box.w * 0.45), int(box.y1 + box.h * 0.5)), col, -1)


# ----------------------------------------------------------------------------- scenarios


def scenario_1() -> Scenario:
    """Rest zone: stops, takes the helmet off, holds it. A static rule says 'no helmet'; the sequence says 'break'."""
    box = BBox(x1=900, y1=380, x2=1010, y2=640)
    steps = [("general", 1.1, 0.0, _ppe(PPEState.WORN, 0.91)), ("rest", 0.6, 0.3, _ppe(PPEState.WORN, 0.90)),
             ("rest", 0.1, 1.8, _ppe(PPEState.WORN, 0.88)), ("rest", 0.0, 2.8, _ppe(PPEState.HELD, 0.78)),
             ("rest", 0.0, 3.8, _ppe(PPEState.HELD, 0.81))]
    frames = [_frame(i, [WorkerFeatures(track_id=7, box=box, zone=z, speed_mps=v, stationary_s=s, torso_tilt_deg=5, facing="toward_camera",
                                        helmet=h, vest=_ppe(PPEState.WORN, 0.9))])
              for i, (z, v, s, h) in enumerate(steps)]
    img = _canvas(); _person(img, box, False)
    return Scenario(1, "휴게구역 진입 후 안전모 탈착", "정상 휴식 (P(yes) 낮음, 오탐 차단)", frames, img)


def scenario_2() -> Scenario:
    """Excavator closing in while the worker has turned away: collision precursor ~2 s ahead."""
    box = BBox(x1=520, y1=380, x2=630, y2=640)
    dist, closing = [5.8, 4.6, 3.4, 2.6, 1.9], [0.0, 1.2, 1.2, 1.2, 1.35]
    frames = []
    for i in range(5):
        eb = BBox(x1=700 + (4 - i) * 70, y1=300, x2=1100 + (4 - i) * 70, y2=640)
        w = WorkerFeatures(track_id=3, box=box, zone="danger" if i >= 2 else "general", speed_mps=0.1, stationary_s=1.5 + i,
                           torso_tilt_deg=6, facing="toward_camera" if i == 0 else "away_from_camera",
                           aware_of_equipment=True if i == 0 else False, helmet=_ppe(PPEState.WORN, 0.9),
                           vest=_ppe(PPEState.WORN, 0.88), nearest_equipment=_rel(1, eb, dist[i], closing[i]))
        frames.append(_frame(i, [w], [EquipmentFeatures(eq_id=1, label="excavator", box=eb, speed_mps=1.2, motion="approaching")]))
    img = _canvas(); _machine(img, frames[-1].equipment[0].box, False); _person(img, box, False)
    return Scenario(2, "굴착기 후진 + 작업자 등 돌림", "협착 전조 (충돌 약 1.4초 전, P(yes) 높음)", frames, img)


def scenario_3() -> Scenario:
    """Work at height: harness not attached, torso leaning off the ladder."""
    box = BBox(x1=300, y1=150, x2=400, y2=420)
    tilt = [4, 6, 9, 15, 22]
    frames = [_frame(i, [WorkerFeatures(track_id=5, box=box, speed_mps=0.1, stationary_s=1.0 + i, torso_tilt_deg=tilt[i], facing="away_from_camera",
                                        helmet=_ppe(PPEState.WORN, 0.9), vest=_ppe(PPEState.WORN, 0.9),
                                        harness=_ppe(PPEState.UNHOOKED, 0.8), on_ladder=True,
                                        climb_height_m=[3.5, 3.5, 3.6, 3.6, 3.7][i], torso_outside_ladder=i == 4)])
              for i in range(5)]
    img = _canvas(); cv2.rectangle(img, (290, 120), (410, 640), (80, 120, 160), 3); _person(img, box, False)
    return Scenario(3, "고소 작업 중 안전고리 미결속 + 상체 기울기", "추락 전조 (P(yes) 높음)", frames, img)


def scenario_4() -> Scenario:
    """Backlit scene: helmet confidence decays to 0.32 while an excavator approaches -> multimodal fallback."""
    box = BBox(x1=640, y1=300, x2=740, y2=590)
    dist, closing, conf = [6.0, 5.2, 4.4, 3.7, 3.1], [0.0, 0.8, 0.8, 0.8, 0.8], [0.62, 0.48, 0.41, 0.35, 0.32]
    frames = []
    for i in range(5):
        eb = BBox(x1=820 + (4 - i) * 50, y1=250, x2=1200 + (4 - i) * 50, y2=600)
        helmet = _ppe(PPEState.WORN if conf[i] >= 0.55 else PPEState.UNCERTAIN, conf[i])
        w = WorkerFeatures(track_id=9, box=box, speed_mps=0.3, stationary_s=0.5, torso_tilt_deg=None, facing="unknown",
                           helmet=helmet, nearest_equipment=_rel(2, eb, dist[i], closing[i]))
        frames.append(_frame(i, [w], [EquipmentFeatures(eq_id=2, label="excavator", box=eb, speed_mps=0.8, motion="approaching")]))
    img = _canvas(backlit=True); cv2.circle(img, (1000, 90), 140, (255, 255, 255), -1)
    _machine(img, frames[-1].equipment[0].box, True); _person(img, box, True)
    return Scenario(4, "역광·가려짐으로 안전모 인식률 0.32 + 장비 접근", "이미지 폴백 자동 트리거 → Jev가 직접 확인", frames, img)


SCENARIOS = {1: scenario_1, 2: scenario_2, 3: scenario_3, 4: scenario_4}


# ----------------------------------------------------------------------------- offline stand-in for the API


def simulated_transport() -> httpx.MockTransport:
    """Local stand-in with the real response schema. Heuristic only; it is NOT the Jev model."""

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        await asyncio.sleep(0.1)
        has_image = bool(body.get("assets"))
        if has_image and req.headers.get(IMAGES_HEADER) != "true":
            return httpx.Response(400, json={"error": f"missing {IMAGES_HEADER} header"})
        if body.get("model") != "jev-1.14.0":
            return httpx.Response(400, json={"error": "unexpected model"})
        p = 0.04
        for w in body["state"]["sequence"]["workers"]:
            m, eq = w["moments"][-1], w["moments"][-1]["nearest_equipment"]
            q = 0.05 if m["zone"] == "rest" and not eq else 0.04
            if eq and eq["ttc_s"] is not None:
                q = 0.94 if (eq["ttc_s"] <= 2.0 and m["aware_of_equipment"] is False) else 0.60 if eq["ttc_s"] <= 4.5 else 0.3
            if m["on_ladder"] and m["harness_status"] != "hooked":
                q = max(q, 0.9 if (m["tilt_deg"] or 0) >= 15 else 0.5)
            if m["helmet_status"].startswith("uncertain") and eq:
                q = 0.86 if has_image else min(q, 0.62)  # an image resolves the doubt, text alone cannot
            p = max(p, q)
        plan = ("equipment_emergency_stop_interlock" if p >= 0.8 else "worker_smartwatch_vibration" if p >= 0.5 else "normal_monitoring")
        probs = {k: 0.02 for k in ACTIONS}
        probs[plan] = 0.94
        answers = {"is_imminent_hazard": {"type": "noul", "noul": p},
                   "risk_score": {"type": "score", "score": round(p * 5, 2), "confidence": 0.7, "probabilities": {}},
                   "action_plan": {"type": "choice", "choice": plan, "confidence": 0.9, "probabilities": probs}}
        if has_image:
            answers["hazard_roi_box"] = {"type": "box", "box": {"top": 0.15, "bottom": 0.92, "left": 0.28, "right": 0.62}, "confidence": 0.8}
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": answers,
                                         "usage": {"input_tokens": 3200 if has_image else 520, "output_tokens": 40}})

    return httpx.MockTransport(handler)


# ----------------------------------------------------------------------------- runner


def redact(body: dict) -> dict:
    """Truncate base64 image data so payloads stay readable on the console."""
    b = copy.deepcopy(body)
    for a in (b.get("assets") or {}).values():
        n = len(a["data_base64"])
        a["data_base64"] = a["data_base64"][:32] + f"...<{n} chars base64 JPEG>"
    return b


def _pretty(body: dict) -> str:
    """Indented JSON with each per-second moment on one line (keeps long payloads readable)."""
    b, slots = redact(body), {}
    for w in b["state"]["sequence"]["workers"]:
        for i, m in enumerate(w["moments"]):
            key = f"@@M{len(slots)}@@"
            slots[key], w["moments"][i] = json.dumps(m, ensure_ascii=False), key
    s = json.dumps(b, ensure_ascii=False, indent=2)
    for k, v in slots.items():
        s = s.replace(f'"{k}"', v)
    return s


def _dump(title: str, headers: dict[str, str], body: dict, st: Settings, name: str, show: bool) -> None:
    """Always saves the full request under out/mock_payloads; prints a readable copy when `show`."""
    safe = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    out = st.out_dir / "mock_payloads"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"headers": safe, "body": body}, ensure_ascii=False, indent=2))
    if show:
        print(f"\n  ┌─ {title}")
        print(f"  │ headers: {json.dumps(safe)}")
        for line in _pretty(body).splitlines():
            print("  │ " + line)


async def run_mock(st: Settings, which: list[int], offline: bool, show_payload: bool = True) -> None:
    live = bool(st.api_key) and not offline
    client = None if live else httpx.AsyncClient(transport=simulated_transport())
    predictor = JevSafetyPredictor(st, client)
    print(f"Jev endpoint: {st.api_url if live else 'OFFLINE stand-in (heuristic responder, not the real Jev)'}  model={st.model}")
    for sid in which:
        sc = SCENARIOS[sid]()
        print(f"\n{'=' * 78}\nScenario {sc.id}: {sc.title}\n  기대 결과: {sc.expect}")
        pipe = SafetyPipeline(st, predictor, SafetyActionRouter(st), verbose=True)
        buf: SequenceBuffer = pipe.buffer
        for f in sc.frames:
            buf.push(f)
        last = sc.frames[-1]
        a = assess(last, st)
        seq = buf.build()
        for w in seq.workers:
            print(f"  worker {w.track_id} narrative: {w.narrative}")
        print(f"  risk reasons: {a.reasons or '-'} | fallback trigger: {a.fallback_worker is not None}")
        text_req, text_h = predictor.build_request(seq, None)
        _dump(f"Scenario {sid} REQUEST (text-only mode)", text_h, text_req.model_dump(mode="json", exclude_none=True),
              st, f"scenario{sid}_text", show_payload)
        roi = None
        if a.fallback_worker is not None:
            roi = prepare_roi(sc.image, focus_box(last, a.fallback_worker))
            img_req, img_h = predictor.build_request(seq, roi)
            _dump(f"Scenario {sid} REQUEST (image-fallback mode)", img_h, img_req.model_dump(mode="json", exclude_none=True),
                  st, f"scenario{sid}_image", show_payload)
            (st.out_dir / "mock_payloads" / f"scenario{sid}_roi.jpg").write_bytes(roi.jpeg)
            print("\n  -- text-only result (for comparison, no action taken) --")
            txt = await predictor.predict(seq, None)
            print(f"  [jev:text] P(hazard)={txt.p_hazard:.2f} risk={txt.risk_score:.1f}/5 plan={txt.action_plan}")
            print("  -- image-fallback result (used for routing) --")
        else:
            print("\n  -- Jev result (text mode) --")
        d = await pipe.decide(seq, roi, a)
        if d.roi_box:
            b = d.roi_box
            print(f"  hazard_roi_box (full-frame px): ({b.x1:.0f},{b.y1:.0f})-({b.x2:.0f},{b.y2:.0f}) conf={d.roi_conf}")
    await predictor.aclose()
