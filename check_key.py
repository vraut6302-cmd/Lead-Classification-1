"""One-minute check that your Sarvam key works for all three models.

Bulbul speaks a Hindi sentence -> Saaras transcribes it back -> Sarvam-105B replies.
    python check_key.py
"""
import base64
import time

from app import config
from app.sarvam import SarvamClient

if not config.SARVAM_API_KEY:
    raise SystemExit("Put SARVAM_API_KEY=... in .env first (copy .env.example).")

c = SarvamClient()
line = "नमस्ते, मुझे अपने 3 BHK फ्लैट के लिए मॉड्यूलर किचन चाहिए।"

t = time.time()
wav = base64.b64decode(c.speak(line, "hi-IN"))
open(config.DATA_DIR / "smoke_tts.wav", "wb").write(wav)
print(f"✓ Bulbul ({config.TTS_MODEL})  {len(wav)//1024} KB audio in {time.time()-t:.1f}s -> data/smoke_tts.wav")

t = time.time()
stt = c.transcribe(wav)
print(f"✓ Saaras ({config.STT_MODEL})  '{stt['transcript']}' [{stt['language_code']}] in {time.time()-t:.1f}s")

t = time.time()
reply = c.chat([{"role": "user", "content": stt["transcript"] or line}], max_tokens=300, reasoning=None)
print(f"✓ Sarvam ({config.CHAT_MODEL})  {reply.strip()[:160]} in {time.time()-t:.1f}s")
