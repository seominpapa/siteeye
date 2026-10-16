"""FastAPI app: REST for control, WebSockets for the live stream and for browser-webcam ingest.

Public mode (--public) is meant to sit behind a Cloudflare Tunnel on 127.0.0.1:
  - one isolated session per visitor (`sid`), with concurrency / per-IP / duration / Jev-call limits
  - no CCTV URLs and no server cameras (the server would connect to arbitrary addresses)
  - nothing is written to disk; WebSockets must come from the site's own origin
"""
from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Polygon, Settings
from ..vision import weights_available
from .demo import DEMO_DIR, DEMOS
from .runner import AnalysisRunner, StartRequest, probe_cameras
from .sessions import Busy, SessionManager

STATIC = Path(__file__).parent / "static"
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
CSP = ("default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class ZonesRequest(BaseModel):
    danger: list[Polygon] = []
    rest: list[Polygon] = []


def client_ip(headers, peer: str | None) -> str:
    """Real visitor IP. Cloudflare's header is only trusted when the request comes from the local tunnel process."""
    if peer in ("127.0.0.1", "::1"):
        return headers.get("cf-connecting-ip") or headers.get("x-forwarded-for", "").split(",")[0].strip() or peer or "local"
    return peer or "unknown"


def create_app(st: Settings, offline: bool = False) -> FastAPI:
    mgr = SessionManager(st, offline)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        if st.public:  # warm one slot plus the shared B/C tracks so the first visitor does not wait for model loading
            ex = await loop.run_in_executor(mgr.pool, mgr.acquire_extractor, "full")
            mgr.release_extractor(ex)
        janitor = asyncio.create_task(mgr.janitor())
        yield
        janitor.cancel()

    app = FastAPI(title="siteeye", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.mgr = mgr

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        resp.headers["Content-Security-Policy"] = CSP
        if st.public:
            resp.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "public, max-age=300"
        else:  # local use: always pick up code changes after an update
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    def sid_of(request: Request) -> str:
        sid = request.query_params.get("sid") or ("local" if not st.public else "")
        if not SID_RE.match(sid) and sid != "local":
            raise HTTPException(400, "missing or invalid session id")
        return sid

    def vid_of(request: Request) -> str:
        """Visitor id (stored in the browser's localStorage) used to remember which demos this visitor already ran."""
        vid = request.query_params.get("vid", "")
        return vid if SID_RE.match(vid) else sid_of(request)

    def session_of(request: Request) -> AnalysisRunner:
        return mgr.session(sid_of(request), client_ip(request.headers, request.client.host if request.client else None))

    def origin_ok(ws: WebSocket) -> bool:
        if not st.public:
            return True
        origin = ws.headers.get("origin", "")
        host = urlparse(origin).netloc
        allowed = urlparse(st.allowed_origin).netloc if st.allowed_origin else ws.headers.get("host", "")
        return bool(origin) and host == allowed

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}  # for tunnel / uptime monitors; deliberately reveals nothing else

    @app.get("/api/info")
    def info(request: Request) -> dict:
        s = session_of(request)
        return {
            "public": st.public, "source_url": st.source_url,
            "limits": {"session_s": st.session_max_s, "max_sessions": st.max_sessions} if st.public else None,
            "jev": {"live": mgr.live_for_new_session(), "model": st.model, "endpoint": st.api_url if mgr.live_for_new_session() and not st.public else None},
            "demos": [{"id": d.id, "title": d.title, "expect": d.expect, "watch": d.watch, "kind": d.kind,
                       "has_video": d.video_file() is not None} for d in DEMOS.values()],
            "precisions": {"fast": True, "accurate": True, "full": weights_available(st.ppe_weights, st.models_dir)},
            "default_precision": "accurate" if st.public else "fast",
            "status": s.status, "demos_done": mgr.demos_done(vid_of(request)),
        }

    @app.post("/api/start")
    async def start(req: StartRequest, request: Request) -> dict:
        s = session_of(request)
        try:
            await s.start(req, client_ip(request.headers, request.client.host if request.client else None), vid_of(request))
        except Busy as exc:
            raise HTTPException(429, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.post("/api/stop")
    async def stop(request: Request) -> dict:
        await session_of(request).stop()
        return {"ok": True}

    @app.post("/api/zones")
    def zones(req: ZonesRequest, request: Request) -> dict:
        session_of(request).set_zones(req.danger, req.rest)
        return {"ok": True}

    if not st.public:  # probing server cameras and serving saved snapshots only make sense on your own machine
        @app.get("/api/cameras")
        async def cameras() -> dict:
            found = await asyncio.get_running_loop().run_in_executor(None, probe_cameras)
            return {"cameras": found, "any": any(c["ok"] for c in found)}

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket) -> None:
        sid = ws.query_params.get("sid") or ("local" if not st.public else "")
        if not origin_ok(ws) or not (SID_RE.match(sid) or sid == "local"):
            await ws.close(code=1008)
            return
        await ws.accept()
        runner = mgr.session(sid, client_ip(ws.headers, ws.client.host if ws.client else None))
        client = runner.add_client(ws)
        try:
            while True:
                await ws.receive_text()  # keep-alive pings from the page
        except WebSocketDisconnect:
            pass
        finally:
            runner.remove_client(client)

    @app.websocket("/ws/ingest")
    async def ws_ingest(ws: WebSocket) -> None:
        sid = ws.query_params.get("sid") or ("local" if not st.public else "")
        if not origin_ok(ws) or not (SID_RE.match(sid) or sid == "local"):
            await ws.close(code=1008)
            return
        await ws.accept()
        runner = mgr.session(sid, client_ip(ws.headers, ws.client.host if ws.client else None))
        try:
            while True:
                runner.push_ingest(await ws.receive_bytes())
        except WebSocketDisconnect:
            pass

    if not st.public:
        snap_dir = st.out_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/snapshots", StaticFiles(directory=snap_dir), name="snapshots")
    app.mount("/demo-media", StaticFiles(directory=DEMO_DIR), name="demo-media")
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
