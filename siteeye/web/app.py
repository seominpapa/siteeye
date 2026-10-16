"""FastAPI app: REST for control, WebSockets for the live stream and for browser-webcam ingest."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Polygon, Settings
from .demo import DEMO_DIR, DEMOS
from .runner import AnalysisRunner, StartRequest, probe_cameras

STATIC = Path(__file__).parent / "static"


class ZonesRequest(BaseModel):
    danger: list[Polygon] = []
    rest: list[Polygon] = []


def create_app(st: Settings, offline: bool = False) -> FastAPI:
    app = FastAPI(title="siteeye", docs_url=None, redoc_url=None)
    runner = AnalysisRunner(st, offline, st.out_dir)
    app.state.runner = runner

    @app.get("/api/info")
    def info() -> dict:
        return {
            "jev": {"live": runner.live, "model": st.model, "endpoint": st.api_url if runner.live else None},
            "demos": [{"id": d.id, "title": d.title, "expect": d.expect, "watch": d.watch, "kind": d.kind} for d in DEMOS.values()],
            "status": runner.status,
        }

    @app.post("/api/start")
    async def start(req: StartRequest) -> dict:
        try:
            await runner.start(req)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.post("/api/stop")
    async def stop() -> dict:
        await runner.stop()
        return {"ok": True}

    @app.post("/api/zones")
    def zones(req: ZonesRequest) -> dict:
        st.danger_zones, st.rest_zones = req.danger, req.rest  # read live by the spatial processor each frame
        return {"ok": True}

    @app.get("/api/cameras")
    async def cameras() -> dict:
        found = await asyncio.get_running_loop().run_in_executor(None, probe_cameras)
        return {"cameras": found, "any": any(c["ok"] for c in found)}

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket) -> None:
        await ws.accept()
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
        await ws.accept()
        try:
            while True:
                runner.push_ingest(await ws.receive_bytes())
        except WebSocketDisconnect:
            pass

    app.mount("/demo-media", StaticFiles(directory=DEMO_DIR), name="demo-media")
    app.mount("/snapshots", StaticFiles(directory=runner.snap_dir), name="snapshots")
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
