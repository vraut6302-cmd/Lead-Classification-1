"""Thin REST client for the three Sarvam APIs this agent uses.

Plain HTTP (httpx) against the documented endpoints, so it doesn't depend on the
SDK's argument names changing between versions.

  Saaras      POST /speech-to-text          (multipart)   speech  -> text + detected language
  Sarvam-105B POST /v1/chat/completions     (OpenAI-style) text   -> reply / structured JSON
  Bulbul      POST /text-to-speech          (JSON)        text    -> base64 WAV
"""
from __future__ import annotations

import json
import logging
import time

import httpx

from . import config

log = logging.getLogger("sarvam")


class SarvamError(RuntimeError):
    pass


class SarvamClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.SARVAM_API_KEY
        self.base = base_url or config.SARVAM_BASE_URL
        self.http = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    # ------------------------------------------------------------------ helpers
    def _headers(self, json_body: bool = True) -> dict:
        h = {"api-subscription-key": self.api_key, "Authorization": f"Bearer {self.api_key}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _post(self, path: str, retries: int = 2, **kw) -> dict:
        url = f"{self.base}{path}"
        for attempt in range(retries + 1):
            r = self.http.post(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise SarvamError(f"{path} -> HTTP {r.status_code}: {r.text[:400]}")
            return r.json()
        raise SarvamError(f"{path} failed after retries")

    # ------------------------------------------------------------------ Saaras
    def transcribe(self, wav_bytes: bytes, language_code: str | None = None) -> dict:
        """Returns {'transcript': str, 'language_code': 'hi-IN' | None}. Max 30 s audio."""
        data = {"model": config.STT_MODEL, "mode": "transcribe"}
        if language_code:
            data["language_code"] = language_code
        if config.STT_MODEL.startswith("saaras:v4") and config.STT_KEYTERMS:
            data["keyterms"] = json.dumps(config.STT_KEYTERMS)
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        try:
            out = self._post("/speech-to-text", data=data, files=files,
                             headers=self._headers(json_body=False))
        except SarvamError as e:
            # Keyterms are v4-only; if the account/model rejects them, retry without.
            if "keyterms" in data and ("422" in str(e) or "400" in str(e)):
                data.pop("keyterms")
                out = self._post("/speech-to-text", data=data, files=files,
                                 headers=self._headers(json_body=False))
            else:
                raise
        return {"transcript": (out.get("transcript") or "").strip(),
                "language_code": out.get("language_code")}

    # ------------------------------------------------------------- Sarvam-105B
    def chat(self, messages: list[dict], model: str | None = None, *, max_tokens: int = 900,
             temperature: float = 0.3, json_mode: bool = False,
             reasoning: str | None = "default") -> str:
        body = {"model": model or config.CHAT_MODEL, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        # reasoning="default" -> use config; None -> explicitly disable thinking (faster).
        eff = config.CHAT_REASONING if reasoning == "default" else reasoning
        body["reasoning_effort"] = eff
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        attempts = [dict(body)]
        # Graceful degradation if an optional field is rejected by the endpoint.
        if json_mode:
            attempts.append({k: v for k, v in body.items() if k != "response_format"})
        attempts.append({k: v for k, v in body.items()
                         if k not in ("response_format", "reasoning_effort")})

        last_err = None
        for b in attempts:
            try:
                out = self._post("/v1/chat/completions", json=b, headers=self._headers())
                msg = out["choices"][0]["message"]
                content = msg.get("content") or ""
                if not content.strip() and out["choices"][0].get("finish_reason") == "length":
                    raise SarvamError("Model used all tokens on reasoning; raise max_tokens.")
                return content
            except SarvamError as e:
                last_err = e
                if not any(code in str(e) for code in ("HTTP 400", "HTTP 422")):
                    raise
                log.warning("chat retry without optional params: %s", e)
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------ Bulbul
    def speak(self, text: str, language_code: str = "en-IN") -> str:
        """Returns base64-encoded WAV."""
        if language_code not in config.TTS_LANGUAGES:
            language_code = "en-IN"
        body = {"text": text[:2400], "language_code": language_code,
                "model": config.TTS_MODEL, "speaker": config.TTS_SPEAKER,
                "pace": config.TTS_PACE, "speech_sample_rate": config.TTS_SAMPLE_RATE,
                "output_audio_codec": "wav"}
        out = self._post("/text-to-speech", json=body, headers=self._headers())
        audios = out.get("audios") or []
        # One input text -> one audio; join defensively as the docs suggest.
        return "".join(audios)
