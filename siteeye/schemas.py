"""Pydantic v2 schemas: single-frame features, the 5-second sequence payload, and the Jev API contract."""
from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, computed_field

# --------------------------------------------------------------------------- geometry & detections


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Ground contact estimate: bottom-center of the box."""
        return (self.x1 + self.x2) / 2, self.y2

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def inter(self, o: BBox) -> float:
        w = min(self.x2, o.x2) - max(self.x1, o.x1)
        h = min(self.y2, o.y2) - max(self.y1, o.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, o: BBox) -> float:
        i = self.inter(o)
        u = self.area + o.area - i
        return i / u if u > 0 else 0.0

    def rows(self, top: float, bottom: float, grow_x: float = 0.0, grow_y: float = 0.0) -> BBox:
        """Sub-box covering fractions [top, bottom] of this box's height, optionally grown."""
        h, w = self.h, self.w
        return BBox(x1=self.x1 - grow_x * w, x2=self.x2 + grow_x * w,
                    y1=self.y1 + top * h - grow_y * h, y2=self.y1 + bottom * h + grow_y * h)

    def union(self, o: BBox) -> BBox:
        return BBox(x1=min(self.x1, o.x1), y1=min(self.y1, o.y1), x2=max(self.x2, o.x2), y2=max(self.y2, o.y2))

    def grown(self, f: float, w_max: float, h_max: float) -> BBox:
        dx, dy = self.w * f, self.h * f
        return BBox(x1=max(0, self.x1 - dx), y1=max(0, self.y1 - dy), x2=min(w_max, self.x2 + dx), y2=min(h_max, self.y2 + dy))


class Keypoint(BaseModel):
    x: float
    y: float
    conf: float


class Detection(BaseModel):
    label: str  # canonical label, see vision.canonical_label
    conf: float
    box: BBox
    source: Literal["A", "B", "C"]
    track_id: int | None = None
    keypoints: list[Keypoint] | None = None  # 17 COCO keypoints (track A only)


# --------------------------------------------------------------------------- per-frame features


class PPEState(str, Enum):
    WORN = "worn"
    ABSENT = "absent"
    HELD = "held_in_hand"
    UNCERTAIN = "uncertain"
    NOT_DETECTED = "not_detected"
    HOOKED = "hooked"
    UNHOOKED = "unhooked"


class PPEItem(BaseModel):
    state: PPEState = PPEState.NOT_DETECTED
    conf: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        """Text for Jev. Borderline detections are kept and tagged, e.g. 'uncertain (conf: 0.38)'."""
        if self.state is PPEState.UNCERTAIN and self.conf is not None:
            return f"uncertain (conf: {self.conf:.2f})"
        return self.state.value


class EquipmentRelation(BaseModel):
    eq_id: int
    label: str
    distance_m: float
    closing_speed_mps: float  # > 0 when the gap is shrinking
    ttc_s: float | None = None  # time-to-collision, None if not closing
    box: BBox | None = Field(default=None, exclude=True)


class WorkerFeatures(BaseModel):
    track_id: int
    box: BBox
    zone: Literal["general", "rest", "danger"] = "general"
    speed_mps: float = 0.0
    gait_variability: float = 0.0  # std/mean of recent speed; high = stumbling/irregular
    stationary_s: float = 0.0
    torso_tilt_deg: float | None = None  # 0 = upright, 90 = horizontal
    facing: Literal["toward_camera", "away_from_camera", "left", "right", "unknown"] = "unknown"
    aware_of_equipment: bool | None = None
    helmet: PPEItem = Field(default_factory=PPEItem)
    vest: PPEItem = Field(default_factory=PPEItem)
    harness: PPEItem = Field(default_factory=PPEItem)
    on_ladder: bool = False
    climb_height_m: float | None = None
    torso_outside_ladder: bool | None = None
    nearest_equipment: EquipmentRelation | None = None


class EquipmentFeatures(BaseModel):
    eq_id: int
    label: str
    box: BBox
    speed_mps: float = 0.0
    motion: Literal["approaching", "receding", "moving", "stationary"] = "stationary"


class FrameFeatures(BaseModel):
    t: float
    frame_idx: int
    frame_wh: tuple[int, int]
    workers: list[WorkerFeatures] = Field(default_factory=list)
    equipment: list[EquipmentFeatures] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    reasons: list[str] = Field(default_factory=list)
    urgent: bool = False  # bypass the periodic Jev cadence
    fallback_worker: int | None = None  # worker whose low-confidence PPE + risky context triggers the image fallback


# --------------------------------------------------------------------------- 5-second sequence payload


class Moment(BaseModel):
    t: str  # "T-4s" ... "T-0s"
    zone: str
    speed_mps: float
    stationary_s: float
    tilt_deg: float | None
    facing: str
    aware_of_equipment: bool | None
    helmet_status: str
    vest_status: str
    harness_status: str
    on_ladder: bool
    climb_height_m: float | None
    torso_outside_ladder: bool | None
    nearest_equipment: EquipmentRelation | None
    events: list[str]


class WorkerTimeline(BaseModel):
    track_id: int
    narrative: str  # causal one-liner: [T-4s: ...] -> [T-2s: ...] -> [T-0s: ...]
    moments: list[Moment]


class EquipmentSummary(BaseModel):
    eq_id: int
    label: str
    motion: str
    speed_mps: float


class SequencePayload(BaseModel):
    camera: str
    window: str
    site_notes: list[str]
    workers: list[WorkerTimeline]
    equipment: list[EquipmentSummary]


# --------------------------------------------------------------------------- Jev API contract


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None  # {"true": ..., "false": ...}


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]  # level descriptions, API allows 2-10


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class BoxQuestion(BaseModel):
    type: Literal["box"] = "box"  # requires jev-1.14 and the images header
    instructions: str


Question = Annotated[Union[NoulQuestion, ScoreQuestion, ChoiceQuestion, BoxQuestion], Field(discriminator="type")]


class ImageAsset(BaseModel):
    type: Literal["image"] = "image"
    format: Literal["jpg", "png"] = "jpg"
    data_base64: str


class JevRequest(BaseModel):
    state: dict
    model: str
    questions: dict[str, Question]
    assets: dict[str, ImageAsset] | None = None


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float  # P(yes)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class BoxEdges(BaseModel):
    top: float
    bottom: float
    left: float
    right: float


class BoxAnswer(BaseModel):
    type: Literal["box"]
    box: BoxEdges
    confidence: float | None = None


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, BoxAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class JevResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)


class JevDecision(BaseModel):
    mode: Literal["text", "image_fallback"]
    p_hazard: float
    risk_score: float  # 0-5
    action_plan: str
    action_probs: dict[str, float] = Field(default_factory=dict)
    roi_box: BBox | None = None  # mapped back to full-frame pixels
    roi_conf: float | None = None
    model: str = ""
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def deg(rad: float) -> float:
    return math.degrees(rad)
