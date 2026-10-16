"""Demo sources for the web app.

Scripted demos replay the four mock scenarios as a rendered 15 fps "video": the scene is drawn from the
scenario's features (interpolated between the 1-second keyframes), so the full pipeline downstream
(sequence buffer, Jev call, routing) behaves exactly as with a camera. No vision model is involved.
The street demo is a real video file that goes through the real YOLO tracks.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..mock import SCENARIOS
from ..schemas import BBox, FrameFeatures, PPEState

W, H, FPS = 1280, 720, 15
HOLD_S = 2.0  # keep the final state on screen before the loop restarts
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"


@dataclass
class DemoInfo:
    id: str
    title: str
    expect: str
    watch: str
    kind: str  # "scripted" | "video"
    zones: dict = field(default_factory=dict)  # normalized polygons, display only
    path: str | None = None


DEMOS: dict[str, DemoInfo] = {d.id: d for d in [
    DemoInfo("s1", "① 휴게구역 안전모 탈착", "정상 휴식으로 판단 (오탐 차단)",
             "안전모가 벗겨졌지만 휴게구역에서 멈춘 맥락이라 P(위험)이 낮게 유지됩니다.", "scripted",
             {"rest": [[0.58, 0.55], [0.92, 0.55], [0.92, 0.97], [0.58, 0.97]]}),
    DemoInfo("s2", "② 굴착기 접근 + 작업자 등 돌림", "협착 전조 → 즉시 조치",
             "굴착기가 다가오는데 작업자는 반대쪽을 보고 서 있습니다. 예상 충돌 시간이 줄며 인터록이 발동됩니다.", "scripted",
             {"danger": [[0.30, 0.50], [0.80, 0.50], [0.80, 0.98], [0.30, 0.98]]}),
    DemoInfo("s3", "③ 고소 작업 안전고리 미결속", "추락 전조 → 즉시 조치",
             "사다리 위에서 안전고리가 걸려 있지 않고 상체가 사다리 밖으로 기울어집니다. 장비가 없어 인터록은 제외됩니다.", "scripted"),
    DemoInfo("s4", "④ 역광으로 안전모 인식 저하", "이미지 폴백 자동 트리거",
             "안전모 인식 신뢰도가 0.32까지 떨어지고 장비가 접근하면, 텍스트 대신 ROI 이미지를 함께 보내 Jev가 직접 확인합니다.", "scripted"),
    DemoInfo("street", "⑤ 실제 영상 (YOLO 실처리)", "실제 모델이 사람을 탐지·추적",
             "버스 앞을 걷는 실제 영상입니다. YOLO 포즈 추적이 돌아가며 안전모·장비는 없어 위험 판단은 나오지 않는 것이 정상입니다.",
             "video", {"rest": [[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]]}, str(DEMO_DIR / "street_bus.mp4")),
]}


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def _lerp_box(a: BBox, b: BBox, f: float) -> BBox:
    return BBox(x1=_lerp(a.x1, b.x1, f), y1=_lerp(a.y1, b.y1, f), x2=_lerp(a.x2, b.x2, f), y2=_lerp(a.y2, b.y2, f))


class ScriptedDemo:
    """One scenario as a looping real-time stream: 4 s of story + HOLD_S on the final state."""

    def __init__(self, sid: str):
        self.sid = int(sid[1:])
        self.frames: list[FrameFeatures] = SCENARIOS[self.sid]().frames
        self.duration = 4.0 + HOLD_S

    def features_at(self, el: float, ts: float) -> FrameFeatures:
        key = min(el, 4.0)
        i = min(int(key), 4)
        frac = key - i if i < 4 else 0.0
        a, b = self.frames[i], self.frames[min(i + 1, 4)]
        f = a.model_copy(deep=True)
        f.t = ts
        for wa, wb, w in zip(a.workers, b.workers, f.workers):
            ra, rb, r = wa.nearest_equipment, wb.nearest_equipment, w.nearest_equipment
            if ra and rb and r:
                r.distance_m = round(_lerp(ra.distance_m, rb.distance_m, frac), 1)
                r.closing_speed_mps = round(_lerp(ra.closing_speed_mps, rb.closing_speed_mps, frac), 2)
                r.ttc_s = round(r.distance_m / r.closing_speed_mps, 1) if r.closing_speed_mps > 0.3 else None
                if ra.box and rb.box:
                    r.box = _lerp_box(ra.box, rb.box, frac)
            if wa.torso_tilt_deg is not None and wb.torso_tilt_deg is not None:
                w.torso_tilt_deg = round(_lerp(wa.torso_tilt_deg, wb.torso_tilt_deg, frac), 1)
            w.stationary_s = round(wa.stationary_s + frac, 1)
        for ea, eb, e in zip(a.equipment, b.equipment, f.equipment):
            e.box = _lerp_box(ea.box, eb.box, frac)
        return f

    async def frames_stream(self, loop: bool):
        """Yields (ts, image, features, restarted) in real time."""
        while True:
            start = time.monotonic()
            first = True
            while (el := time.monotonic() - start) < self.duration:
                ts = time.time()
                f = self.features_at(el, ts)
                yield ts, self.render(f), f, first
                first = False
                await asyncio.sleep(max(0.0, 1 / FPS - (time.monotonic() - start - el)))
            if not loop:
                return

    # ------------------------------------------------------------------ rendering

    def render(self, f: FrameFeatures) -> np.ndarray:
        backlit = self.sid == 4
        top, bottom = ((250, 235, 215), (170, 160, 150)) if backlit else ((215, 190, 160), (150, 140, 130))
        img = np.empty((H, W, 3), np.uint8)
        img[:] = np.linspace(top, bottom, H)[:, None, :].astype(np.uint8)
        cv2.rectangle(img, (0, 560), (W, H), (92, 98, 104), -1)
        for x in range(0, W, 160):  # ground markings
            cv2.line(img, (x, 600), (x - 90, H), (110, 116, 122), 2)
        if backlit:
            cv2.circle(img, (1010, 90), 150, (255, 255, 255), -1)
            img = cv2.addWeighted(img, 0.78, np.full_like(img, 255), 0.22, 0)
        for w in f.workers:
            if w.on_ladder:
                b = w.box
                lx1, lx2 = int(b.x1 - 20), int(b.x2 + 20)
                for y in range(int(b.y1) - 30, 640, 40):
                    cv2.line(img, (lx1, y), (lx2, y), (70, 110, 150), 3)
                cv2.line(img, (lx1, int(b.y1) - 30), (lx1, 640), (70, 110, 150), 5)
                cv2.line(img, (lx2, int(b.y1) - 30), (lx2, 640), (70, 110, 150), 5)
        for e in f.equipment:
            self._machine(img, e.box, backlit)
        for w in f.workers:
            self._person(img, w, backlit)
        cv2.putText(img, f"DEMO  scenario {self.sid}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)
        return img

    @staticmethod
    def _machine(img: np.ndarray, b: BBox, dark: bool) -> None:
        body, cab = ((40, 40, 45), (35, 35, 40)) if dark else ((35, 170, 235), (30, 140, 200))
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(img, (x1, y1 + int(b.h * 0.40)), (x2, y2 - int(b.h * 0.18)), body, -1)
        cv2.rectangle(img, (x1, y1 + int(b.h * 0.08)), (x1 + int(b.w * 0.42), y1 + int(b.h * 0.45)), cab, -1)
        cv2.rectangle(img, (x1, y2 - int(b.h * 0.18)), (x2, y2), (45, 45, 48), -1)  # tracks
        cv2.line(img, (x1 + int(b.w * 0.40), y1 + int(b.h * 0.30)), (x1 - 40, y1 - 10), cab, 12)  # boom

    @staticmethod
    def _person(img: np.ndarray, w, dark: bool) -> None:
        b = w.box
        x, y, bw, bh = int(b.x1), int(b.y1), int(b.w), int(b.h)
        skin, shirt, pants = ((22, 22, 26),) * 3 if dark else ((190, 160, 130), (60, 110, 200), (70, 60, 50))
        cx, hr = x + bw // 2, max(10, bh // 9)
        cv2.rectangle(img, (x + bw // 6, y + bh // 5), (x + 5 * bw // 6, y + int(bh * 0.62)), shirt, -1)
        cv2.rectangle(img, (x + bw // 5, y + int(bh * 0.62)), (x + 4 * bw // 5, y + bh), pants, -1)
        cv2.circle(img, (cx, y + hr + 2), hr, skin, -1)
        helmet = w.helmet.state
        hc = (30, 215, 250) if not dark else (28, 28, 32)
        if helmet in (PPEState.WORN, PPEState.UNCERTAIN):
            cv2.ellipse(img, (cx, y + hr), (hr + 3, hr + 2), 0, 180, 360, hc, -1)
        elif helmet is PPEState.HELD:
            cv2.ellipse(img, (x - 8, y + int(bh * 0.5)), (hr, hr // 2 + 3), 0, 180, 360, hc, -1)
        if w.on_ladder and w.harness.state is PPEState.UNHOOKED:  # lanyard dangling
            cv2.line(img, (cx, y + int(bh * 0.4)), (cx + 12, y + int(bh * 0.75)), (40, 40, 220), 3)
