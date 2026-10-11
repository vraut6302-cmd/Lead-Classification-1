# Voice Lead Qualifier — a multilingual sales agent on Sarvam AI

Built for a home-interiors company. Interiors brands get thousands of "get a free quote" form leads.
Most are never called in the first golden hour, sales reps burn time on people who are just browsing,
and many customers would rather talk in Hindi, Tamil or Hinglish than English.

This agent calls the lead within minutes, talks naturally in the customer's language, qualifies
them (city, home, scope, budget, timeline, decision maker), books a free design consultation,
and hands the sales team a scored lead card + CRM note.

The company name is configurable (`BRAND` in `.env`, default placeholder "Acme Interiors"), so the
same agent can be white-labelled for any interiors, real-estate or home-services business.

![Call screen with live lead card](docs/screenshot.png)

## How it works

```
 customer speaks ─► Saaras v4 (STT, auto language detect, keyterms: BHK, possession…)
                        │  "3 BHK hai, keys mil gayi"  [hi-IN]
                        ▼
                Sarvam-105B-conversations  ── returns JSON every turn:
                  { reply, reply_language, extracted:{bhk:3, property_status:…}, action }
                        │
          ┌─────────────┼───────────────────────────┐
          ▼             ▼                           ▼
   Bulbul v3 (TTS)   lead state + deterministic   actions: book_slot (validated
   speaks the reply  score (Hot/Warm/Cold/DQ)     against calendar) / end_call
                                                    │
                                     end of call ─► Sarvam-105B writes CRM note ─► leads.jsonl / CSV
```

Design choices worth calling out:

* **The LLM extracts, Python scores.** The score is a transparent rule set in `app/scoring.py`
  (timeline, budget vs. scope, readiness, decision maker, serviceability), so every lead shows *why* it is Hot,
  and the rules can be tuned from real conversion data without touching prompts.
* **Two Sarvam-105B variants.** `sarvam-105b-conversations` with thinking switched off for low-latency turns;
  flagship `sarvam-105b` (reasoning on) for the end-of-call summary.
* **Language mirroring.** Saaras' detected language is passed to the model each turn; it replies in the same
  language/code-mix and Bulbul speaks it in that language.
* **Guardrails.** Discloses it's a virtual assistant, asks if it's a good time, handles callbacks and "not interested",
  never quotes prices, only books slots that exist.
* **Simulation before launch.** `simulate.py` plays 5 customer personas (Hinglish hot lead, family-dependent warm lead,
  explorer, unserviceable city, annoyed customer) against the agent and checks the tier it assigns.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # paste your key from dashboard.sarvam.ai -> API Keys
python check_key.py          # Bulbul -> Saaras -> Sarvam-105B round trip, proves the key works
uvicorn app.server:app --reload
```

Open **http://localhost:8000**, allow the microphone, press **Start call** and play the customer.
Speak in any supported Indian language; the mic stops when you pause (hands-free mode keeps the call flowing).
You can also type. The **Leads** tab lists every finished call with an **Export CSV** button.

No key yet? Leave `SARVAM_API_KEY` empty: the app runs in **mock mode** (rule-based brain, browser voice) so you can
see the whole flow.

```bash
python simulate.py            # persona stress test (uses real Sarvam models when a key is set)
pytest -q                     # unit tests, no key needed
```

### What it costs on the free ₹100 credits

Rough estimates at list prices (Sarvam-105B ₹15 / ₹60 per 1M input / output tokens, speech-to-text ₹30 per hour,
Bulbul ₹30 per 10K characters):

| Run | Approx. cost |
| --- | --- |
| `check_key.py` | well under ₹1 |
| One full voice call (~8 turns) | ₹3–5 — text-to-speech is most of it |
| One typed call in the web UI (still spoken by Bulbul) | ₹3–4 |
| `simulate.py`, all 5 personas (text only, no audio) | ₹3–6 |

So ₹100 covers roughly 20 voice test calls plus a few simulator runs. Build and debug in mock mode,
and spend credits on real calls and the demo recording.

## Files

| Path | What it does |
| --- | --- |
| `app/sarvam.py` | REST client for Saaras (`/speech-to-text`), Sarvam-105B (`/v1/chat/completions`), Bulbul (`/text-to-speech`) with retries and graceful fallbacks |
| `app/agent.py` | System prompt, per-turn JSON protocol, field normalisation, slot booking, end-of-call summary |
| `app/scoring.py` | Explainable lead score, tiers, next best action, serviceable-city check |
| `app/server.py` | FastAPI: `/api/session`, `/api/turn` (audio or text), `/api/leads`, `/api/leads.csv` |
| `static/index.html` | Call UI: mic → 16 kHz WAV, auto-stop on silence, barge-in, live lead card, leads table |
| `app/mock.py` | Offline brain/voice for demos and tests |
| `simulate.py` | Customer-persona simulator with pass/fail report |

## Taking it to production

* **Telephony:** put the same pipeline behind Exotel/Twilio via Sarvam's LiveKit or Pipecat integrations, and switch to
  streaming STT/TTS for sub-second turns.
* **CRM:** replace `store.save_lead()` with LeadSquared / Salesforce, and `make_slots()` with the designer calendar.
* **WhatsApp follow-up:** send the booking confirmation and design catalogue right after the call.
* **Tune the score:** back-fit the weights in `scoring.py` on which leads actually converted.
* Keep the serviceable-city list and budget minimums in `config.py` / `scoring.py` in sync with the business.
