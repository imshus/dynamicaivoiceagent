"""Serves the page and bridges the browser WebSocket to a VoiceSession.

    python server.py            →  http://localhost:8100

Browser → server: binary frames of 16 kHz int16 mono PCM, plus JSON text
frames {"type": "start", "instructions": "..."} and {"type": "stop"}.
Server → browser: binary frames of 24 kHz int16 mono PCM, plus JSON events
ready / user / agent / retract / clear / error.
"""
import asyncio
import json
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent import AGENT_NAMES, DEFAULT_GENDER, VoiceSession, load_prompt

load_dotenv()
for _stream in (sys.stdout, sys.stderr):      # Hindi transcripts in the Windows console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("server")

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PAGE = os.path.join(HERE, "ui", "index.html")
HTTP_PORT = int(os.getenv("PORT", os.getenv("HTTP_PORT", "8100")))
# Where the agent is reachable from outside (Pratham AI). The socket clients
# connect to is /ws on that host, over wss when the site is https.
PUBLIC_URL = (os.getenv("PRATHAM_AI_URL") or os.getenv("PUBLIC_URL") or "").strip().rstrip("/")
SOCKET_URL = ("" if not PUBLIC_URL else
              PUBLIC_URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws")

app = FastAPI(title="Dynamic Voice Agent")
# The page may be hosted elsewhere and still read /prompt here.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/")
async def index():
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/prompt")
async def prompt():
    return {"prompt": load_prompt(), "gender": DEFAULT_GENDER, "names": AGENT_NAMES,
            "public_url": PUBLIC_URL, "socket_url": SOCKET_URL}


@app.websocket("/ws")
async def talk(ws: WebSocket):
    await ws.accept()
    lock = asyncio.Lock()
    closed = False

    async def send_audio(pcm: bytes):
        nonlocal closed
        if closed:
            return
        try:
            async with lock:
                await ws.send_bytes(pcm)
        except Exception:
            closed = True

    async def send_json(payload: dict):
        nonlocal closed
        if closed:
            return
        try:
            async with lock:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            closed = True

    session: VoiceSession | None = None
    logger.info("Browser connected")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            pcm = msg.get("bytes")
            if pcm:
                if session is not None:
                    await session.feed_audio(pcm)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                ctrl = json.loads(text)
            except Exception:
                continue
            kind = ctrl.get("type")
            if kind == "start" and session is None:
                session = VoiceSession(send_audio, send_json, ctrl.get("instructions"), ctrl.get("gender"))
                await session.start()
            elif kind == "interrupt":
                # The client heard its user start talking over the agent.
                if session is not None:
                    await session.interrupt(str(ctrl.get("reason") or "caller spoke")[:60])
            elif kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Session error: {e!r}")
    finally:
        closed = True
        if session is not None:
            await session.close()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Browser disconnected")


if __name__ == "__main__":
    logger.info(f"Dynamic Voice Agent on http://localhost:{HTTP_PORT}"
                + (f" — public {PUBLIC_URL}, socket {SOCKET_URL}" if PUBLIC_URL else ""))
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")
