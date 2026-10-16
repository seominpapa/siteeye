"""Per-visitor sessions and the limits that keep a public demo affordable and safe.

Local mode  : one shared extractor cache, no limits (behaves like the original single-user app).
Public mode : every visitor gets an isolated session; concurrent sessions, per-IP starts, session length and
              Jev calls (per session and per day) are capped. Each concurrent session owns a separate YOLO
              instance, because ByteTrack keeps its state inside the model: sharing one would mix visitors' track ids.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TYPE_CHECKING

from ..config import Settings
from ..vision import PRECISION_TRACKS, HybridVisionExtractor, VisionModelRegistry, pick_device

log = logging.getLogger("siteeye.sessions")

if TYPE_CHECKING:
    from .runner import AnalysisRunner


class Busy(Exception):
    """A limit was hit; the message is shown to the visitor (HTTP 429)."""


class SessionManager:
    def __init__(self, st: Settings, offline: bool):
        self.st, self.offline, self.public = st, offline, st.public
        self.pool = ThreadPoolExecutor(max_workers=1)  # one inference thread shared by all sessions
        self.sessions: dict[str, AnalysisRunner] = {}
        self._ip_active: dict[str, str] = {}
        self._ip_starts: dict[str, deque] = defaultdict(deque)
        self._demo_done: OrderedDict[str, set[str]] = OrderedDict()  # visitor id -> demos already run
        self._day, self.day_calls = date.today(), 0
        self._lock = threading.Lock()
        self._free_a: list = []  # idle pose models (one per concurrent session)
        self._built = 0
        self._aux: dict[str, object] = {}  # shared tracks B / C (None = unavailable)

    # ------------------------------------------------------------------ sessions

    def session(self, sid: str, ip: str) -> AnalysisRunner:
        from .runner import AnalysisRunner

        s = self.sessions.get(sid)
        if s is None:
            s = self.sessions[sid] = AnalysisRunner(self, sid)
        s.ip, s.last_seen = ip, time.monotonic()
        return s

    def active_count(self, slot: bool) -> int:
        """Running sessions that use a YOLO slot (slot=True) or the lightweight scripted demos (slot=False)."""
        return sum(1 for s in self.sessions.values() if s.running and s.uses_slot == slot)

    # ------------------------------------------------------------------ "every visitor can run each demo once"

    def demos_done(self, vid: str) -> list[str]:
        return sorted(self._demo_done.get(vid, ()))

    def mark_demo(self, vid: str, demo_id: str) -> None:
        self._demo_done.setdefault(vid, set()).add(demo_id)
        self._demo_done.move_to_end(vid)
        while len(self._demo_done) > 5000:  # bounded memory
            self._demo_done.popitem(last=False)

    def admit(self, s: AnalysisRunner, ip: str, uses_slot: bool, first_demo: bool) -> None:
        """Called before a session starts. Raises Busy with a visitor-facing message.

        A demo this visitor has not run yet (`first_demo`) skips the per-IP rules, so webcam retries or
        other people behind the same address can never lock a visitor out of seeing each demo once.
        """
        if not self.public:
            return
        if not first_demo:
            other = self._ip_active.get(ip)
            if other and other != s.sid and (o := self.sessions.get(other)) and o.running:
                raise Busy("이 네트워크에서 이미 진행 중인 데모가 있습니다. 먼저 종료해 주세요.")
        if uses_slot and self.active_count(True) >= self.st.max_sessions:
            raise Busy(f"지금 영상·웹캠 분석 접속자가 많습니다 (동시 {self.st.max_sessions}명까지). 잠시 후 다시 시도해 주세요.")
        if not uses_slot and self.active_count(False) >= self.st.max_demo_sessions:
            raise Busy("지금 데모 접속자가 많습니다. 잠시 후 다시 시도해 주세요.")
        if not first_demo:
            now, q = time.time(), self._ip_starts[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.st.ip_sessions_per_hour:
                raise Busy(f"시간당 {self.st.ip_sessions_per_hour}회 한도에 도달했습니다. 아직 보지 않은 데모는 계속 실행할 수 있습니다.")
            q.append(now)
            self._ip_active[ip] = s.sid

    async def janitor(self) -> None:
        """Drop sessions nobody is using any more."""
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for sid, s in list(self.sessions.items()):
                if not s.running and not s.clients and now - s.last_seen > 120:
                    del self.sessions[sid]
            self._ip_active = {ip: sid for ip, sid in self._ip_active.items() if sid in self.sessions and self.sessions[sid].running}

    # ------------------------------------------------------------------ Jev budget

    def _roll_day(self) -> None:
        if date.today() != self._day:
            self._day, self.day_calls = date.today(), 0

    def budget_ok(self) -> bool:
        self._roll_day()
        return self.day_calls < self.st.daily_jev_calls

    def count_call(self) -> None:
        self._roll_day()
        self.day_calls += 1

    def live_for_new_session(self) -> bool:
        return bool(self.st.api_key) and not self.offline and (not self.public or self.budget_ok())

    # ------------------------------------------------------------------ vision extractors (call from the pool thread)

    def _aux_track(self, key: str):
        """Tracks B and C are stateless: built once, then shared by every session (all inference runs on one thread)."""
        if key not in self._aux:
            self._aux[key] = VisionModelRegistry.build_track(key, self.st, pick_device(self.st.device))
        return self._aux[key]

    def acquire_extractor(self, precision: str) -> HybridVisionExtractor:
        """Per-session extractor: its own pose model + tracker (track A) and the shared B/C tracks for the chosen precision."""
        keys = PRECISION_TRACKS.get(precision, "A").split(",")
        with self._lock:
            slot = self._free_a.pop() if self._free_a else None
            if slot is None:
                if self.public and self._built >= self.st.max_sessions:
                    raise Busy("처리 슬롯이 모두 사용 중입니다. 잠시 후 다시 시도해 주세요.")
                self._built += 1
        try:
            if slot is None:
                slot = VisionModelRegistry.build_track("A", self.st, pick_device(self.st.device))
                if slot is None:
                    raise RuntimeError("pose weights could not be loaded")
            tracks = {"A": slot}
            for k in keys[1:]:
                t = self._aux_track(k)
                if t is not None:
                    tracks[k] = t
        except Exception:
            with self._lock:
                self._built -= 1
            raise
        ex = HybridVisionExtractor(tracks, self.st.aux_every, self.st.label_min_conf)
        ex.reset()  # fresh ByteTrack state for the new visitor
        log.info("extractor ready: tracks %s (requested %s)", ex.active, precision)
        return ex

    def release_extractor(self, ex: HybridVisionExtractor | None) -> None:
        if ex is not None and ex.slot is not None:
            with self._lock:
                self._free_a.append(ex.slot)
