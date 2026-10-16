"""Decision layer: scene state + events in, alerts out.

Anything with a `decide(state, events) -> list[Alert]` method works, so a
different decider (e.g. an LLM/decision-model call) can replace RuleDecider
without touching the vision pipeline.
"""
import time
from dataclasses import dataclass


@dataclass
class Alert:
    level: str  # "info" | "warn" | "critical"
    message: str
    track_id: int = -1


class RuleDecider:
    def __init__(self, watch=("person",), zone_only=True, max_count=None, cooldown_s=30.0):
        self.watch = set(watch)
        self.zone_only = zone_only
        self.max_count = max_count or {}  # {"person": 5}
        self.cooldown_s = cooldown_s
        self._last = {}

    def decide(self, state, events):
        alerts = self._rules(state, events)
        now = time.time()
        fresh = []
        for a in alerts:  # one alert per distinct message per cooldown window
            if now - self._last.get(a.message, 0.0) >= self.cooldown_s:
                self._last[a.message] = now
                fresh.append(a)
        return fresh

    def _rules(self, state, events):
        alerts = []
        for e in events:
            if e.name not in self.watch:
                continue
            if e.kind == "enter" and (e.in_zone or not self.zone_only):
                alerts.append(Alert("warn", f"{e.name} #{e.track_id} entered", e.track_id))
            elif e.kind == "dwell":
                alerts.append(Alert("critical", f"{e.name} #{e.track_id} dwelling {e.dwell_s:.0f}s in zone", e.track_id))
        for name, limit in self.max_count.items():
            if state["counts"].get(name, 0) >= limit:
                alerts.append(Alert("warn", f"{name} count {state['counts'][name]} >= {limit}"))
        return alerts
