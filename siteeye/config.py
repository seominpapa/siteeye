"""Runtime settings. Everything tunable lives here; secrets come only from the environment."""
import os
from pathlib import Path

from pydantic import BaseModel, Field

Polygon = list[tuple[float, float]]  # normalized (0-1) image coordinates

# The first ten are the target vocabulary. The last two are descriptive variants: on the demo footage YOLO-World v2
# scored "excavator" ~0.1 but "heavy construction vehicle" 0.4-0.7, and "yellow hard hat" 0.5-0.7 vs "hard hat" ~0.1.
# vision.canonical_label maps them back to the shared labels.
DEFAULT_WORLD_PROMPTS = [
    "person", "hard hat", "safety vest", "safety harness lanyard", "excavator",
    "forklift", "crane hook", "ladder", "scaffolding", "open manhole",
    "yellow hard hat", "heavy construction vehicle", "safety helmet", "orange safety vest", "high visibility vest",
]

DEFAULT_SITE_NOTES = [
    "Taking off a hard hat inside a rest zone is normal behaviour, not a hazard by itself.",
    "Distances are monocular estimates (person height = 1.7 m), accuracy roughly +/-30%.",
    "A status of 'uncertain (conf: x)' means the vision model was unsure; use the surrounding time sequence to judge.",
    "A helmet/vest/harness status of 'not_detected' only means the vision model did not find the item (back view, "
    "backlight, occlusion, small size). It does NOT mean the worker has none. 'not_measured' means no PPE model was "
    "running. Do not conclude that PPE is missing without other evidence.",
    "The harness status from the vision model is unreliable; judge it from the image whenever one is provided.",
    "A machine with equipment.reliable=false is at a very different depth than the worker, so its distance is not trustworthy.",
]


class Settings(BaseModel):
    # --- Jev / TypeSafe ---
    api_url: str = "https://api.typesafe.ai/v1/systemone"  # documented endpoint (override with TYPESAFE_API_URL)
    api_key: str | None = Field(default=None, repr=False)
    model: str = "jev-1.14.0"  # pinned: jev-latest still maps to 1.13 and Box questions need 1.14
    jev_timeout_s: float = 8.0
    jev_retries: int = 1
    jev_interval_s: float = 1.0  # periodic call cadence
    jev_urgent_min_gap_s: float = 0.3  # floor between calls even for urgent events

    # --- vision tracks ---
    tracks: str = "A,B,C"  # which tracks to run; unavailable ones are skipped with a warning
    models_dir: Path = Path("models")
    pose_weights: str = "yolo11m-pose.pt"
    world_weights: str = "yolov8s-worldv2.pt"
    ppe_weights: str = "custom_ppe.pt"
    world_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_WORLD_PROMPTS))
    device: str | None = None
    imgsz: int = 640
    aux_every: int = 3  # run tracks B/C every Nth frame, reuse the last result in between

    # --- confidence bands ---
    conf_floor: float = 0.25  # below this a detection is dropped
    conf_confident: float = 0.55  # [floor, confident) is tagged "uncertain"
    fallback_conf: float = 0.45  # risky context + a PPE/object conf below this triggers the image fallback
    # descriptive prompts ("heavy construction vehicle") fire on buses and trucks at 0.25-0.44, so machinery needs more
    label_min_conf: dict[str, float] = Field(default_factory=lambda: {"heavy machinery": 0.45})

    # --- geometry ---
    person_height_m: float = 1.7
    # A machine of known real height that looks much smaller than that height predicts is farther away than the worker
    # (measured: excavator/worker pixel-height ratio 2.0 at the same depth, 0.92 when it stood ~2x farther).
    depth_ratio_max: float = 1.5
    ppe_flag_p: float = 0.7  # Jev image answer: P(lanyard unattached) >= this, or P(helmet on) <= 1 - this, raises a review
    near_distance_m: float = 4.0
    ttc_warn_s: float = 4.0
    ttc_urgent_s: float = 3.0
    danger_zones: list[Polygon] = Field(default_factory=list)
    rest_zones: list[Polygon] = Field(default_factory=list)

    # --- sequence window ---
    camera: str = "cam1"
    window_s: int = 5
    snapshot_interval_s: float = 1.0
    site_notes: list[str] = Field(default_factory=lambda: list(DEFAULT_SITE_NOTES))

    # --- public hosting (python main.py --web --public) ---
    public: bool = False
    max_sessions: int = 2  # concurrent webcam/video sessions; one inference slot (YOLO + its own tracker) each
    max_demo_sessions: int = 8  # scripted demos need no YOLO slot, so they get their own, larger cap
    session_max_s: int = 180
    max_jev_calls_per_session: int = 200
    daily_jev_calls: int = 4000  # after this, new sessions use the offline stand-in (labelled as such)
    ip_sessions_per_hour: int = 6
    source_url: str = "https://github.com/seominpapa/siteeye"
    allowed_origin: str | None = None  # e.g. https://demo.example.com ; default: same host as the request

    # --- action routing ---
    p_immediate: float = 0.80
    p_review: float = 0.50
    risk_immediate: float = 4.0
    out_dir: Path = Path("out")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ
        kw: dict = {"api_key": env.get("TYPESAFE_API_KEY")}
        for field, var in [
            ("api_url", "TYPESAFE_API_URL"), ("tracks", "SITEEYE_TRACKS"), ("device", "SITEEYE_DEVICE"),
            ("pose_weights", "SITEEYE_POSE_WEIGHTS"), ("world_weights", "SITEEYE_WORLD_WEIGHTS"),
            ("ppe_weights", "SITEEYE_PPE_WEIGHTS"), ("camera", "SITEEYE_CAMERA"),
            ("public", "SITEEYE_PUBLIC"), ("max_sessions", "SITEEYE_MAX_SESSIONS"),
            ("max_demo_sessions", "SITEEYE_MAX_DEMO_SESSIONS"),
            ("session_max_s", "SITEEYE_SESSION_SECONDS"), ("daily_jev_calls", "SITEEYE_DAILY_JEV_CALLS"),
            ("max_jev_calls_per_session", "SITEEYE_SESSION_JEV_CALLS"),
            ("ip_sessions_per_hour", "SITEEYE_IP_SESSIONS_PER_HOUR"),
            ("source_url", "SITEEYE_SOURCE_URL"), ("allowed_origin", "SITEEYE_PUBLIC_ORIGIN"),
        ]:
            if env.get(var):
                kw[field] = env[var]
        return cls(**kw)


def load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    """Minimal .env reader (KEY=VALUE per line). Variables already set in the real environment win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_polygon(text: str) -> Polygon:
    """'0.1,0.5;0.9,0.5;0.9,1;0.1,1' -> [(0.1, 0.5), ...]"""
    return [tuple(float(v) for v in p.split(",")) for p in text.split(";")]  # type: ignore[misc]
