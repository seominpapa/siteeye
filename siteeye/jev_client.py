"""Async Jev 1.14 decision client with a two-track call strategy.

text mode      (default)   only the 5-second sequence JSON: ~100 ms, a few hundred tokens
image fallback (rare)      risky context + low-confidence PPE -> the same state plus an ROI crop, the
                           `TypeSafe-Danger-Zone-Images: true` header, and a Box question so Jev 1.14 can look itself

API facts (https://docs.typesafe.ai/api.md, /images.md, /primitives/box.md):
  POST /v1/systemone, images as base64 `assets` referenced from `state`, 280-1568 px per side,
  >= 78,400 total pixels, Box needs jev-1.14.x. Images/Box are labelled "Dreamer-ready" (experimental).
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from .config import Settings
from .schemas import (BBox, BoxAnswer, BoxQuestion, ChoiceAnswer, ChoiceQuestion, ImageAsset, JevDecision, JevRequest,
                      JevResponse, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion, SequencePayload, clamp)

IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
ACTIONS = {  # ASCII keys for the API, Korean names for operators
    "normal_monitoring": "정상_모니터링",
    "worker_smartwatch_vibration": "작업자_스마트워치_진동경고",
    "site_speaker_announcement": "현장_지향성스피커_경고방송",
    "equipment_emergency_stop_interlock": "중장비_비상정지_인터록_발동",
}
MIN_SIDE, MAX_SIDE, MIN_PIXELS, MAX_PIXELS = 280, 1568, 78_400, 2_458_624


@dataclass
class RoiImage:
    """JPEG bytes of the evidence crop plus the transform needed to map Box answers back to the full frame."""
    jpeg: bytes
    out_w: int
    out_h: int
    origin: tuple[float, float]  # crop top-left in frame pixels
    scale: float  # crop px -> output px
    pad: tuple[int, int]  # left/top letterbox padding in output px


def prepare_roi(frame: np.ndarray, box: BBox, margin: float = 0.25) -> RoiImage:
    """Crop around `box` and fit the image into the API limits (side 280-1568 px, 78,400+ total pixels)."""
    h, w = frame.shape[:2]
    g = box.grown(margin, w, h)
    x1, y1, x2, y2 = int(g.x1), int(g.y1), max(int(g.x2), int(g.x1) + 2), max(int(g.y2), int(g.y1) + 2)
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = max(1.0, MIN_SIDE / min(cw, ch), (MIN_PIXELS / (cw * ch)) ** 0.5)  # upscale small crops
    scale = min(scale, MAX_SIDE / max(cw, ch), (MAX_PIXELS / (cw * ch)) ** 0.5) if (
        max(cw, ch) * scale > MAX_SIDE or cw * ch * scale * scale > MAX_PIXELS) else scale  # ...but stay under the caps
    ow, oh = max(1, round(cw * scale)), max(1, round(ch * scale))
    img = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    tw, th = max(ow, MIN_SIDE), max(oh, MIN_SIDE)  # extreme aspect ratios: letterbox up to 280 px
    pad_l, pad_t = (tw - ow) // 2, (th - oh) // 2
    if (tw, th) != (ow, oh):
        canvas = np.full((th, tw, 3), 114, np.uint8)
        canvas[pad_t:pad_t + oh, pad_l:pad_l + ow] = img
        img = canvas
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return RoiImage(buf.tobytes(), tw, th, (float(x1), float(y1)), scale, (pad_l, pad_t))


def build_questions(image_ref: str | None) -> dict:
    qs: dict = {
        "is_imminent_hazard": NoulQuestion(
            instructions=("최근 5초간의 이동 궤적과 작업자 행동의 개연성을 볼 때, 향후 3초 이내에 협착·추락·실신 등 "
                          "실제 사고로 이어질 전조 증상인가? (휴게구역 내 단순 안전모 탈착 등은 제외)"),
            criteria={"true": "시간 순서상 장비 접근·추락 위험·쓰러짐 등 사고로 이어지는 전조가 이어지고 있다.",
                      "false": "정상 작업 또는 휴식이며 사고로 이어질 연속적인 전조가 없다."}),
        "risk_score": ScoreQuestion(
            instructions="현장의 즉각적인 위험 심각도 점수",
            criteria=["위험 없음", "경미한 주의 필요", "주의: 안전수칙 위반 또는 근접 상황", "경계: 위험이 커지는 중",
                      "위험: 수 초 내 사고 가능성이 높음", "긴급: 사고 직전"]),
        "action_plan": ChoiceQuestion(
            instructions="현재 상황에 가장 적절한 즉시 조치",
            criteria={"normal_monitoring": "정상 모니터링 유지",
                      "worker_smartwatch_vibration": "해당 작업자 스마트워치 진동 경고",
                      "site_speaker_announcement": "현장 지향성 스피커 경고 방송",
                      "equipment_emergency_stop_interlock": "중장비 비상정지 인터록 발동"}),
    }
    if image_ref:
        qs["hazard_roi_box"] = BoxQuestion(
            instructions=f"가장 위험에 노출된 작업자 또는 미착용 안전장구의 위치 (image: {image_ref})")
    return qs


class JevSafetyPredictor:
    def __init__(self, st: Settings, client: httpx.AsyncClient | None = None):
        self.st = st
        self._own = client is None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(st.jev_timeout_s))

    async def aclose(self) -> None:
        if self._own:
            await self.client.aclose()

    # ------------------------------------------------------------------ request building

    def build_request(self, seq: SequencePayload, roi: RoiImage | None) -> tuple[JevRequest, dict[str, str]]:
        state: dict = {"sequence": seq.model_dump(mode="json")}
        assets = None
        headers = {"Content-Type": "application/json"}
        ref = None
        if roi is not None:
            ref = str(uuid.uuid4())
            state["evidence_image"] = ref  # every asset must be referenced somewhere in the request
            assets = {ref: ImageAsset(format="jpg", data_base64=base64.b64encode(roi.jpeg).decode())}
            headers[IMAGES_HEADER] = "true"
        req = JevRequest(state=state, model=self.st.model, questions=build_questions(ref), assets=assets)
        return req, headers

    # ------------------------------------------------------------------ call

    async def predict(self, seq: SequencePayload, roi: RoiImage | None = None) -> JevDecision:
        req, headers = self.build_request(seq, roi)
        if self.st.api_key:
            headers["Authorization"] = f"Bearer {self.st.api_key}"
        body = req.model_dump(mode="json", exclude_none=True)
        t0 = time.perf_counter()
        raw = await self._post(body, headers)
        resp = JevResponse.model_validate(raw)
        return self._decision(resp, roi, (time.perf_counter() - t0) * 1000)

    async def _post(self, body: dict, headers: dict[str, str]) -> dict:
        for attempt in range(self.st.jev_retries + 1):
            try:
                r = await self.client.post(self.st.api_url, json=body, headers=headers)
            except httpx.TransportError as exc:
                if attempt < self.st.jev_retries:
                    await asyncio.sleep(0.4)
                    continue
                raise RuntimeError(f"Jev connection failed: {exc}") from None
            if r.status_code in RETRY_STATUS and attempt < self.st.jev_retries:
                await asyncio.sleep(min(float(r.headers.get("retry-after", 0.5)), 2.0))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ response mapping

    def _decision(self, resp: JevResponse, roi: RoiImage | None, latency_ms: float) -> JevDecision:
        a = resp.answers
        hazard, score, plan, box = a.get("is_imminent_hazard"), a.get("risk_score"), a.get("action_plan"), a.get("hazard_roi_box")
        if not isinstance(hazard, NoulAnswer):
            raise RuntimeError("Jev response is missing is_imminent_hazard")
        roi_box, roi_conf = None, None
        if isinstance(box, BoxAnswer) and roi is not None:
            e = box.box  # fractions of the sent image -> full-frame pixels
            to_frame = lambda x, y: (roi.origin[0] + (x * roi.out_w - roi.pad[0]) / roi.scale,  # noqa: E731
                                     roi.origin[1] + (y * roi.out_h - roi.pad[1]) / roi.scale)
            (x1, y1), (x2, y2) = to_frame(e.left, e.top), to_frame(e.right, e.bottom)
            roi_box, roi_conf = BBox(x1=x1, y1=y1, x2=x2, y2=y2), box.confidence
        return JevDecision(
            mode="image_fallback" if roi is not None else "text", p_hazard=clamp(hazard.noul, 0, 1),
            risk_score=clamp(score.score, 0, 5) if isinstance(score, ScoreAnswer) else 0.0,
            action_plan=plan.choice if isinstance(plan, ChoiceAnswer) else "normal_monitoring",
            action_probs=plan.probabilities if isinstance(plan, ChoiceAnswer) else {},
            roi_box=roi_box, roi_conf=roi_conf, model=resp.model, latency_ms=latency_ms, usage=resp.usage)
