"""CLI: python -m siteeye --source rtsp://... [--show] [--save out.mp4] [--events events.jsonl]"""
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

from .decision import RuleDecider
from .detector import Detector
from .events import SceneTracker, Zone
from .jev import DEFAULT_QUESTIONS, DryRunTransport, JevDecider, JevTransport
from .source import FrameSource


def parse_args():
    p = argparse.ArgumentParser(prog="siteeye")
    p.add_argument("--source", required=True, help="RTSP/HTTP URL, video file, or webcam index")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO weights (downloaded on first use)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default=None, help="mps | cuda:0 | cpu (default: auto)")
    p.add_argument("--classes", nargs="*", help="only detect these classes, e.g. person car")
    p.add_argument("--zone", help='normalized polygon "x,y;x,y;..." for zone/dwell logic')
    p.add_argument("--dwell", type=float, default=10.0, help="seconds in zone before a dwell alert")
    p.add_argument("--alert-classes", nargs="*", default=["person"])
    p.add_argument("--max-count", nargs="*", default=[], help='alert when class count >= n, e.g. person=5')
    p.add_argument("--jev", action="store_true", help="add Jev judgement (needs TYPESAFE_API_KEY)")
    p.add_argument("--jev-dry-run", action="store_true", help="print the text state that would go to Jev; no network")
    p.add_argument("--jev-model", default="jev-1.14.0", help="pinned model id (jev-latest currently maps to 1.13)")
    p.add_argument("--jev-threshold", type=float, default=0.7, help="alert when P(statement is true) >= this")
    p.add_argument("--jev-url", help="override the API endpoint (for tests)")
    p.add_argument("--jev-interval", type=float, default=10.0, help="min seconds between periodic Jev calls")
    p.add_argument("--question", action="append", help="statement for Jev to judge true/false (repeatable; default: built-in set)")
    p.add_argument("--camera", default="cam1", help="camera name included in the state text")
    p.add_argument("--show", action="store_true", help="display annotated stream (q to quit)")
    p.add_argument("--save", help="write annotated video to this path")
    p.add_argument("--events", help="append events/alerts as JSON lines to this path")
    p.add_argument("--loop", action="store_true", help="loop a video file")
    p.add_argument("--no-pace", action="store_true", help="read files as fast as possible")
    return p.parse_args()


def color(track_id):
    hsv = np.uint8([[[(track_id * 47) % 180, 220, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def annotate(frame, dets, zone, state, fps, lat_ms):
    h, w = frame.shape[:2]
    if zone is not None:
        cv2.polylines(frame, [zone.pixels((w, h))], True, (0, 200, 255), 2)
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        c = color(d.track_id if d.track_id >= 0 else d.cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"{d.name} #{d.track_id} {d.conf:.2f}" if d.track_id >= 0 else f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    counts = " ".join(f"{k}:{v}" for k, v in sorted(state["counts"].items())) or "-"
    hud = f"{fps:4.1f} fps  {lat_ms:4.0f} ms  |  {counts}"
    cv2.rectangle(frame, (0, 0), (8 + 9 * len(hud), 24), (0, 0, 0), -1)
    cv2.putText(frame, hud, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    a = parse_args()
    det = Detector(a.weights, a.imgsz, a.conf, a.classes, a.device)
    zone = Zone.parse(a.zone) if a.zone else None
    scene = SceneTracker(zone=zone, dwell_s=a.dwell)
    decider = RuleDecider(
        watch=a.alert_classes,
        zone_only=zone is not None,
        max_count={k: int(v) for k, v in (s.split("=") for s in a.max_count)},
    )
    if a.jev or a.jev_dry_run:
        transport = DryRunTransport() if a.jev_dry_run else JevTransport(
            url=a.jev_url, model=a.jev_model, threshold=a.jev_threshold)
        decider = JevDecider(transport, base=decider, camera=a.camera, interval_s=a.jev_interval,
                             questions=a.question or DEFAULT_QUESTIONS)
    src = FrameSource(a.source, pace_files=not a.no_pace, loop=a.loop).start()
    log = open(a.events, "a", buffering=1) if a.events else None
    writer, seq, stamps, lat_ms = None, 0, deque(maxlen=30), 0.0
    dropped = n_proc = 0
    print(f"device={det.device} weights={a.weights} source={a.source}")
    try:
        while True:
            got = src.read(seq)
            if got is None:
                if src.ended:
                    break
                print("waiting for frames...")
                continue
            dropped += got[0] - seq - 1 if seq else 0
            seq, ts, frame = got
            n_proc += 1
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            dets = det(frame)
            lat_ms = (time.perf_counter() - t0) * 1000
            events = scene.update(dets, ts, (w, h))
            state = scene.state(ts)
            alerts = decider.decide(state, events)
            stamps.append(time.perf_counter())
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0

            for e in events:
                print(f"[event] {e.kind:5s} {e.name} #{e.track_id}")
            for al in alerts:
                print(f"[ALERT:{al.level}] {al.message}")
            if log:
                for rec in [{"type": "event", **e.__dict__} for e in events] + \
                           [{"type": "alert", "t": ts, **al.__dict__} for al in alerts]:
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if a.show or a.save:
                vis = annotate(frame, dets, zone, state, fps, lat_ms)
                if a.save:
                    if writer is None:
                        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src.fps or 25.0, (w, h))
                    writer.write(vis)
                if a.show:
                    cv2.imshow("siteeye", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if dropped and n_proc % 100 == 0:
                print(f"[warn] detector behind stream: {dropped} frames dropped so far ({fps:.1f} fps)")
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        print(f"done: {n_proc} frames processed, {dropped} dropped")
        if writer:
            writer.release()
        if log:
            log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
