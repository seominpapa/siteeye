"""Construction-site near-miss prediction.

  python main.py --mock                      # 4 scripted scenarios, no camera/GPU (offline stand-in unless TYPESAFE_API_KEY is set)
  python main.py --source site.mp4           # video file, RTSP URL or webcam index
  python main.py --web                       # local web app with CCTV/webcam input and demo scenarios
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):  # checked before the package imports: they use 3.10 syntax
    sys.exit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}). "
             "Create the virtualenv with a newer interpreter, e.g. python3.12 -m venv .venv")

import argparse
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx

from siteeye.config import Settings, parse_polygon
from siteeye.jev_client import JevSafetyPredictor
from siteeye.mock import SCENARIOS, run_mock, simulated_transport
from siteeye.pipeline import SafetyPipeline
from siteeye.router import SafetyActionRouter
from siteeye.schemas import FrameFeatures
from siteeye.source import FrameSource
from siteeye.spatial import SpatialFeatureProcessor
from siteeye.vision import VisionModelRegistry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="run the scripted scenarios")
    mode.add_argument("--source", help="video file, RTSP/HTTP URL or webcam index")
    mode.add_argument("--web", action="store_true", help="local web app (open http://127.0.0.1:8000)")
    p.add_argument("--public", action="store_true", help="web: public demo mode for a Cloudflare Tunnel (limits, no CCTV URLs, no storage)")
    p.add_argument("--host", default="127.0.0.1", help="web: bind address (keep 127.0.0.1 unless you add authentication)")
    p.add_argument("--port", type=int, default=8000, help="web: port")
    p.add_argument("--scenario", default="all", help="mock scenario 1-4 or all")
    p.add_argument("--offline", action="store_true", help="never call the real API (local stand-in responder)")
    p.add_argument("--tracks", help="vision tracks, e.g. A,B,C (env SITEEYE_TRACKS)")
    p.add_argument("--device", help="mps | cuda:0 | cpu")
    p.add_argument("--camera", help="camera name placed in the Jev state")
    p.add_argument("--danger-zone", action="append", default=[], help='polygon "x,y;x,y;..." (0-1), repeatable')
    p.add_argument("--rest-zone", action="append", default=[], help='polygon "x,y;x,y;..." (0-1), repeatable')
    p.add_argument("--jev-interval", type=float, help="seconds between periodic Jev calls (default 1.0)")
    p.add_argument("--show", action="store_true", help="display annotated stream (q to quit)")
    p.add_argument("--loop", action="store_true", help="loop a video file")
    p.add_argument("--no-pace", action="store_true", help="read files as fast as possible (timestamps compress)")
    p.add_argument("--log", help="append decisions as JSON lines")
    p.add_argument("--quiet-payload", action="store_true", help="mock: do not print request payloads")
    return p.parse_args()


def build_settings(a: argparse.Namespace) -> Settings:
    st = Settings.from_env()
    if a.tracks:
        st.tracks = a.tracks
    if a.device:
        st.device = a.device
    if a.camera:
        st.camera = a.camera
    if a.jev_interval:
        st.jev_interval_s = a.jev_interval
    st.danger_zones = [parse_polygon(z) for z in a.danger_zone]
    st.rest_zones = [parse_polygon(z) for z in a.rest_zone]
    return st


def draw(frame, f: FrameFeatures, pipe: SafetyPipeline) -> None:
    for w in f.workers:
        b = w.box
        col = (0, 0, 255) if w.zone == "danger" else (0, 200, 0)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, 2)
        rel = w.nearest_equipment
        txt = f"#{w.track_id} helmet:{w.helmet.status}" + (f" {rel.label} {rel.distance_m:.1f}m" if rel else "")
        cv2.putText(frame, txt, (int(b.x1), max(14, int(b.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for e in f.equipment:
        cv2.rectangle(frame, (int(e.box.x1), int(e.box.y1)), (int(e.box.x2), int(e.box.y2)), (0, 165, 255), 2)
        cv2.putText(frame, f"{e.label} {e.motion}", (int(e.box.x1), int(e.box.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    if pipe.last:
        d = pipe.last
        cv2.putText(frame, f"jev {d.mode} P={d.p_hazard:.2f} risk={d.risk_score:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


async def run_stream(st: Settings, a: argparse.Namespace) -> None:
    if not st.api_key and not a.offline:
        sys.exit("TYPESAFE_API_KEY is not set. Export it, or pass --offline to use the local stand-in responder.")
    client = httpx.AsyncClient(transport=simulated_transport()) if (a.offline or not st.api_key) else None
    predictor = JevSafetyPredictor(st, client)
    pipe = SafetyPipeline(st, predictor, SafetyActionRouter(st), log_path=a.log)
    extractor = VisionModelRegistry.build(st)
    proc = SpatialFeatureProcessor(st, ppe_model=any(k in extractor.tracks for k in ("B", "C")))
    print(f"tracks={extractor.active} source={a.source} jev={'offline stand-in' if client else st.api_url}")

    def process(frame, ts: float, idx: int) -> FrameFeatures:
        h, w = frame.shape[:2]
        return proc.process(extractor.infer(frame, idx), (w, h), ts, idx)

    src = FrameSource(a.source, pace_files=not a.no_pace, loop=a.loop).start()
    loop, pool = asyncio.get_running_loop(), ThreadPoolExecutor(max_workers=1)  # one inference thread; event loop stays free for Jev
    seq = idx = 0
    try:
        while True:
            got = await loop.run_in_executor(None, src.read, seq)
            if got is None:
                if src.ended:
                    break
                continue
            seq, ts, frame = got
            feats = await loop.run_in_executor(pool, process, frame, ts, idx)
            await pipe.on_frame(feats, frame)
            idx += 1
            if a.show:
                draw(frame, feats, pipe)
                cv2.imshow("siteeye", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        src.stop()
        await pipe.drain()
        await predictor.aclose()
        cv2.destroyAllWindows()
        print(f"done: {idx} frames")


def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    st = build_settings(a)
    if a.web:
        import uvicorn

        from siteeye.web.app import create_app

        if a.public:
            st.public = True
            st.jev_urgent_min_gap_s = max(st.jev_urgent_min_gap_s, 1.0)  # keep API spend predictable
            if a.host != "127.0.0.1":
                print("WARNING: public mode should listen on 127.0.0.1 and be exposed only through the tunnel")
        live = bool(st.api_key) and not a.offline
        print(f"siteeye web app: http://{a.host}:{a.port}   Jev: {'LIVE ' + st.model if live else 'offline stand-in (set TYPESAFE_API_KEY for the real model)'}"
              + (f"   PUBLIC mode: {st.max_sessions} concurrent x {st.session_max_s}s, {st.daily_jev_calls} Jev calls/day" if st.public else ""))
        uvicorn.run(create_app(st, a.offline), host=a.host, port=a.port, log_level="warning", ws_max_size=1_000_000)
    elif a.mock:
        which = sorted(SCENARIOS) if a.scenario == "all" else [int(a.scenario)]
        asyncio.run(run_mock(st, which, a.offline, not a.quiet_payload))
    else:
        asyncio.run(run_stream(st, a))


if __name__ == "__main__":
    main()
