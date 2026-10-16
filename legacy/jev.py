"""Jev decision layer: YOLO scene state -> text -> Jev -> alerts.

Only text leaves the machine (class names, counts, durations); no frames are sent.
Calls run on a worker thread so the 70-500 ms Jev latency never stalls the frame loop.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .decision import Alert

# Jev evaluates statements (yes/no probability), so these are phrased as claims, not questions.
DEFAULT_QUESTIONS = (
    "A person appears to be intruding or loitering in the restricted zone.",
    "The number of people or vehicles is unusual for a camera like this.",
)


@dataclass
class Answer:
    question: str
    verdict: bool  # True = the condition in the question holds
    detail: str = ""
    level: str = "warn"


def build_state_text(camera, state, history, now):
    """Compact plain-text description of the scene for the decision model."""
    stamp = lambda t: datetime.fromtimestamp(t).strftime("%H:%M:%S")
    counts = ", ".join(f"{n} x{c} ({state['in_zone'].get(n, 0)} in zone)" for n, c in sorted(state["counts"].items()))
    lines = [f"camera: {camera}", f"time: {stamp(now)}", f"objects now: {counts or 'none'}"]
    if state["tracks"]:
        lines.append("tracked objects:")
        for t in state["tracks"]:
            where = f"in zone for {t['zone_s']:.0f}s" if t["in_zone"] else "outside zone"
            lines.append(f"- {t['name']} #{t['id']}: visible {t['age_s']:.0f}s, {where}")
    if history:
        lines.append("recent events:")
        for t, e in history:
            extra = f" after {e.dwell_s:.0f}s" if e.kind in ("exit", "dwell") and e.dwell_s else ""
            lines.append(f"- {stamp(t)} {e.name} #{e.track_id} {e.kind}{extra}{' (zone)' if e.in_zone else ''}")
    return "\n".join(lines)


class JevTransport:
    """TypeSafe System One endpoint (https://docs.typesafe.ai/api): one `noul` (yes/no) question per statement.

    Each statement is judged against the state text and comes back as the probability that it is true;
    probability >= `threshold` counts as a positive verdict. The key is read from TYPESAFE_API_KEY, never from code.
    """

    URL = "https://api.typesafe.ai/v1/systemone"
    RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

    def __init__(self, api_key=None, url=None, model="jev-1.14.0", timeout=5.0, threshold=0.7, retries=1):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("set TYPESAFE_API_KEY (or use --jev-dry-run to inspect the payload)")
        self.url, self.model = url or self.URL, model
        self.timeout, self.threshold, self.retries = timeout, threshold, retries

    def ask(self, state_text, questions):
        body = {
            "state": state_text,
            "model": self.model,
            "questions": {f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate(questions)},
        }
        answers = self._post(body).get("answers", {})
        out = []
        for i, q in enumerate(questions):
            ans = answers.get(f"q{i}")
            if not ans or ans.get("type") != "noul":
                continue
            p = float(ans["noul"])
            out.append(Answer(q, p >= self.threshold, f"p={p:.2f}", "critical" if p >= 0.95 else "warn"))
        return out

    def _post(self, body):
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": "siteeye/0.1"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in self.RETRY_STATUS and attempt < self.retries:
                    time.sleep(min(float(e.headers.get("Retry-After") or 0.5), 2.0))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read(300).decode(errors='replace')}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(0.5)
                    continue
                raise RuntimeError(f"connection failed: {e}") from None


class DryRunTransport:
    """Prints what would be sent and answers nothing; for inspecting the payload offline."""

    def ask(self, state_text, questions):
        print("[jev dry-run] state:\n" + "\n".join("    " + ln for ln in state_text.splitlines()))
        print("[jev dry-run] questions: " + " | ".join(questions))
        return []


class JevDecider:
    """Wraps a fast base decider (rules) and adds Jev's judgement without blocking the caller."""

    def __init__(self, transport, base=None, camera="cam1", questions=DEFAULT_QUESTIONS,
                 interval_s=10.0, history_s=30.0, cooldown_s=60.0):
        self.transport, self.base, self.camera = transport, base, camera
        self.questions = list(questions)
        self.interval_s, self.history_s, self.cooldown_s = interval_s, history_s, cooldown_s
        self._history = deque()
        self._last_call = 0.0
        self._last_alert = {}
        self._busy = threading.Event()
        self._lock = threading.Lock()
        self._ready = []  # Alerts produced by finished calls, drained by decide()

    def decide(self, state, events):
        now = time.time()
        alerts = self.base.decide(state, events) if self.base else []
        for e in events:
            self._history.append((now, e))
        while self._history and now - self._history[0][0] > self.history_s:
            self._history.popleft()

        due = events or (state["counts"] and now - self._last_call >= self.interval_s)
        if due and not self._busy.is_set():  # latest-wins: never queue behind a slow call
            self._busy.set()
            self._last_call = now
            text = build_state_text(self.camera, state, list(self._history), now)
            threading.Thread(target=self._call, args=(text,), daemon=True).start()

        with self._lock:
            alerts += self._ready
            self._ready = []
        return alerts

    def _call(self, text):
        try:
            answers = self.transport.ask(text, self.questions)
        except Exception as exc:  # network/API failure must not take down detection
            answers = []
            print(f"[jev] call failed: {exc}")
        now = time.time()
        fresh = []
        for a in answers:
            msg = f"jev: {a.question} -> {a.detail or 'yes'}"
            if a.verdict and now - self._last_alert.get(a.question, 0.0) >= self.cooldown_s:
                self._last_alert[a.question] = now
                fresh.append(Alert(a.level, msg))
        with self._lock:
            self._ready += fresh
        self._busy.clear()
