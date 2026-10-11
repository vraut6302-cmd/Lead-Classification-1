"""FastAPI server: voice turn pipeline (Saaras -> Sarvam-105B -> Bulbul) + lead APIs + web UI."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, store
from .agent import LeadAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

if config.MOCK_MODE:
    from .mock import MockBrain, MockVoice
    brain, voice = MockBrain(), MockVoice()
    log.warning("Running in MOCK MODE (no SARVAM_API_KEY). Scripted brain, no real STT/TTS.")
else:
    from .sarvam import SarvamClient
    brain = voice = SarvamClient()

agent = LeadAgent(brain, voice)
app = FastAPI(title="Voice Lead Qualifier (Sarvam)")
STATIC = config.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _tts(text: str, lang: str) -> tuple[str, float]:
    t = time.perf_counter()
    try:
        audio = voice.speak(text, lang) if text else ""
    except Exception as e:
        log.error("TTS failed: %s", e)
        audio = ""
    return audio, round(time.perf_counter() - t, 2)


class StartBody(BaseModel):
    name: str | None = None
    city: str | None = None
    locality: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "mock": config.MOCK_MODE, "brand": config.BRAND, "agent": config.AGENT_NAME, "chat_model": config.CHAT_MODEL,
            "stt_model": config.STT_MODEL, "tts_model": config.TTS_MODEL,
            "speaker": config.TTS_SPEAKER}


@app.post("/api/session")
def start_session(body: StartBody):
    t = time.perf_counter()
    try:
        sess, turn = agent.start(body.model_dump())
    except Exception as e:
        log.exception("start failed")
        raise HTTPException(502, f"Agent error: {e}")
    llm_s = round(time.perf_counter() - t, 2)
    audio, tts_s = _tts(turn["reply"], turn["reply_language"])
    return {**sess.snapshot(), "reply": turn["reply"], "reply_language": turn["reply_language"],
            "audio_b64": audio, "timings": {"llm": llm_s, "tts": tts_s}}


@app.post("/api/turn")
async def turn(session_id: str = Form(...), text: str | None = Form(None),
               language: str | None = Form(None), audio: UploadFile | None = File(None)):
    if session_id not in agent.sessions:
        raise HTTPException(404, "Unknown session — start a new call.")
    timings = {}
    user_lang = language or None

    # 1) Speech -> text (Saaras)
    if audio is not None:
        wav = await audio.read()
        if len(wav) < 2000:
            raise HTTPException(400, "Recording too short.")
        t = time.perf_counter()
        try:
            stt = voice.transcribe(wav)
        except Exception as e:
            log.exception("STT failed")
            raise HTTPException(502, f"Speech-to-text failed: {e}")
        timings["stt"] = round(time.perf_counter() - t, 2)
        text, user_lang = stt["transcript"], stt["language_code"] or user_lang
        if not text:
            return {"no_speech": True, "timings": timings,
                    **agent.sessions[session_id].snapshot()}
    if not text or not text.strip():
        raise HTTPException(400, "Send either audio or text.")

    # 2) Think (Sarvam-105B) + update lead state + score
    t = time.perf_counter()
    try:
        result = agent.turn(session_id, text.strip(), user_lang)
    except Exception as e:
        log.exception("agent turn failed")
        raise HTTPException(502, f"Agent error: {e}")
    timings["llm"] = round(time.perf_counter() - t, 2)

    # 3) Text -> speech (Bulbul)
    audio_b64, timings["tts"] = _tts(result["reply"], result.get("reply_language", "en-IN"))

    sess = agent.sessions[session_id]
    return {**sess.snapshot(), "user_text": text, "user_language": user_lang,
            "reply": result["reply"], "reply_language": result.get("reply_language"),
            "updates": result.get("updates", {}), "action": result.get("action"),
            "booking": result.get("booking"), "audio_b64": audio_b64,
            "timings": timings, "lead_id": sess.lead_id}


@app.post("/api/session/{session_id}/end")
def end_session(session_id: str):
    sess = agent.sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Unknown session")
    agent.finish(sess)
    return {**sess.snapshot(), "lead_id": sess.lead_id}


@app.get("/api/leads")
def leads():
    return store.list_leads()


@app.get("/api/leads.csv", response_class=PlainTextResponse)
def leads_csv():
    return PlainTextResponse(store.leads_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=leads.csv"})
