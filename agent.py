"""The lead-qualification agent: conversation state, prompt, per-turn logic, booking, wrap-up."""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import config, scoring, store
from .scoring import ENUMS

log = logging.getLogger("agent")
IST = timezone(timedelta(hours=5, minutes=30))

LEAD_FIELDS = ["name", "city", "locality", "property_type", "bhk", "property_status",
               "scope", "budget_lakh", "start_timeline", "decision_maker", "style_preference",
               "consultation_mode", "consultation_slot", "callback_time", "not_interested",
               "notes"]


# ----------------------------------------------------------------------------- slots
def make_slots(now: datetime | None = None) -> list[dict]:
    """Next 3 days x 3 times. In production, read the designer calendar instead."""
    now = now or datetime.now(IST)
    slots, n = [], 1
    day = now.date()
    while len({s["date"] for s in slots}) < 3:
        day += timedelta(days=1)
        for hour in (11, 15, 18):
            dt = datetime(day.year, day.month, day.day, hour, tzinfo=IST)
            label = dt.strftime("%a %d %b, ") + dt.strftime("%I %p").lstrip("0")
            slots.append({"id": f"S{n}", "date": str(day), "label": label})
            n += 1
    return slots


# ----------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You are {agent}, a friendly virtual assistant calling on behalf of {brand}, a home-interiors company.
The customer recently enquired about home interiors on our website. Your job on this short call:
qualify the lead, then book a free design consultation with a {brand} designer.

HOW TO TALK (this is a phone call, your text is converted to speech):
- 1–2 short sentences per turn. Ask ONE question at a time. Warm, natural, never pushy.
- Reply in the customer's language. If they mix Hindi/English (Hinglish) or another code-mix, mirror it.
  Their detected language is given as [lang: xx-IN] before each message. Use simple everyday words.
- Say money as words a person would say: "8 lakh", "12 lakh", "1.2 crore". No symbols, no lists, no emojis, no markdown.
- First turn: introduce yourself as {brand}'s virtual assistant, confirm you are speaking to the customer, ask if it's a good time.
- If it's a bad time, ask for a callback time (save callback_time) and end politely.
- If they are not interested or ask not to be called, apologise, set not_interested=true and end the call.
- Never invent prices, discounts or delivery dates. If asked, say the designer will share an exact quote after seeing the floor plan.
  You may say {brand} offers end-to-end design and execution and the first consultation is free.

INFORMATION TO COLLECT (skip anything already known; follow roughly this order):
city (and locality), property_type, bhk, property_status (keys received / possession soon / under construction / renovation),
scope (full home, kitchen + wardrobes, only kitchen, only wardrobes, specific rooms), start_timeline, budget_lakh,
decision_maker (do they decide alone or with spouse/family — if family, suggest they join the consultation).
Optional if it comes up naturally: style_preference.

BOOKING: once the key details are known (or the customer is keen), offer TWO convenient slots from AVAILABLE SLOTS and ask
how they'd like to meet: experience_centre visit, home_visit, or virtual. When they pick, use action "book_slot" with the slot_id
and set consultation_mode, confirm the slot in your reply, say a designer will WhatsApp them, then wrap up.
If the customer's city is not serviceable, politely say we aren't there yet, offer to notify them, and end the call.

CURRENT KNOWN LEAD DETAILS (JSON): {lead}
STILL MISSING: {missing}
CITY SERVICEABLE: {serviceable}
AVAILABLE SLOTS: {slots}
CURRENT TIME: {now}

OUTPUT FORMAT — respond with ONLY one JSON object, no other text:
{{"reply": "<what you say next>",
  "reply_language": "<BCP-47 code of your reply, e.g. hi-IN, en-IN, ta-IN>",
  "extracted": {{<ONLY fields learned or corrected in the customer's LAST message>}},
  "action": "none" | "book_slot" | "end_call",
  "slot_id": "<S1..S9 when action is book_slot, else null>"}}

Allowed values for "extracted":
name: string | city: string | locality: string | bhk: integer | budget_lakh: number (in lakh INR; 1 crore = 100)
property_type: {property_type}
property_status: {property_status}
scope: {scope}
start_timeline: {start_timeline}   (within_1m = within a month, 1_3m, 3_6m, 6m_plus, exploring = just looking)
decision_maker: {decision_maker}
consultation_mode: {consultation_mode}
style_preference: string | callback_time: string | not_interested: boolean | notes: string (anything else useful for the designer)
Use "end_call" only for your final goodbye turn."""


def build_system_prompt(lead: dict, slots: list[dict]) -> str:
    known = {k: v for k, v in lead.items() if v not in (None, "", [], False)}
    svc = scoring.is_serviceable(lead.get("city"))
    return SYSTEM_PROMPT.format(
        agent=config.AGENT_NAME, brand=config.BRAND,
        lead=json.dumps(known, ensure_ascii=False),
        missing=", ".join(scoring.missing_fields(lead)) or "nothing — move to booking",
        serviceable={True: "yes", False: "NO", None: "unknown yet"}[svc],
        slots="; ".join(f"{s['id']}={s['label']}" for s in slots),
        now=datetime.now(IST).strftime("%A %d %B %Y, %I:%M %p IST"),
        **{k: " | ".join(v) for k, v in ENUMS.items()},
    )


# ----------------------------------------------------------------------------- parsing
def parse_model_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    # Model answered in plain text: treat it as the spoken reply.
    return {"reply": t, "extracted": {}, "action": "none"}


def normalise(extracted: dict) -> dict:
    """Coerce types and drop values outside the allowed enums."""
    out = {}
    for k, v in (extracted or {}).items():
        if k not in LEAD_FIELDS or v in (None, "", "null"):
            continue
        if k in ENUMS:
            v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
            v = {"experience_center": "experience_centre"}.get(v, v)
            if v not in ENUMS[k]:
                log.info("dropping %s=%r (not an allowed value)", k, v)
                continue
        elif k == "bhk":
            m = re.search(r"\d+", str(v))
            if not m:
                continue
            v = int(m.group(0))
        elif k == "budget_lakh":
            try:
                v = float(re.sub(r"[^\d.]", "", str(v)))
            except ValueError:
                continue
            if v > 10000:          # model gave rupees instead of lakh
                v = v / 100000
            v = round(v, 2)
        elif k == "not_interested":
            v = v is True or str(v).lower() in {"true", "yes", "1"}
        out[k] = v
    return out


# ----------------------------------------------------------------------------- session
@dataclass
class Session:
    id: str
    lead: dict
    slots: list[dict]
    history: list[dict] = field(default_factory=list)   # chat messages for the LLM
    transcript: list[dict] = field(default_factory=list)  # for the UI / CRM
    language: str = "en-IN"
    done: bool = False
    summary: str = ""
    lead_id: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(IST).isoformat())

    def snapshot(self) -> dict:
        sc = scoring.score_lead(self.lead, final=self.done)
        return {"session_id": self.id, "lead": self.lead, "score": sc.to_dict(),
                "missing": scoring.missing_fields(self.lead), "done": self.done,
                "summary": self.summary, "language": self.language,
                "slots": self.slots}


class LeadAgent:
    def __init__(self, brain, voice):
        """brain: object with .chat(messages, **kw) -> str ; voice: STT/TTS (may be None)."""
        self.brain = brain
        self.voice = voice
        self.sessions: dict[str, Session] = {}

    # -- lifecycle -------------------------------------------------------------
    def start(self, prefill: dict | None = None) -> tuple[Session, dict]:
        lead = {k: None for k in LEAD_FIELDS}
        lead["not_interested"] = False
        for k, v in (prefill or {}).items():
            if k in ("name", "city", "locality") and v:
                lead[k] = str(v).strip()
        sess = Session(id=uuid.uuid4().hex[:10], lead=lead, slots=make_slots())
        self.sessions[sess.id] = sess
        # The agent speaks first on an outbound call.
        turn = self._agent_turn(sess, user_text=None)
        return sess, turn

    def turn(self, session_id: str, user_text: str, user_lang: str | None) -> dict:
        sess = self.sessions[session_id]
        if sess.done:
            return {"reply": "", "action": "end_call"}
        if user_lang:
            sess.language = user_lang
        sess.transcript.append({"role": "customer", "text": user_text, "lang": user_lang,
                                "t": datetime.now(IST).isoformat()})
        return self._agent_turn(sess, user_text=user_text, user_lang=user_lang)

    # -- core ---------------------------------------------------------------
    def _agent_turn(self, sess: Session, user_text: str | None, user_lang: str | None = None) -> dict:
        if user_text is not None:
            sess.history.append({"role": "user",
                                 "content": f"[lang: {user_lang or sess.language}] {user_text}"})
        else:
            sess.history.append({"role": "user", "content":
                                 "[call connected — the customer picked up and said 'Hello?']"})

        messages = [{"role": "system", "content": build_system_prompt(sess.lead, sess.slots)}]
        messages += sess.history
        raw = self.brain.chat(messages, json_mode=True, max_tokens=900, temperature=0.4,
                              reasoning=None)
        data = parse_model_json(raw)

        updates = normalise(data.get("extracted") or {})
        sess.lead.update(updates)

        action = (data.get("action") or "none").strip().lower()
        booking = None
        if action == "book_slot":
            slot = next((s for s in sess.slots if s["id"] == str(data.get("slot_id")).upper()), None)
            if slot:
                sess.lead["consultation_slot"] = slot["label"]
                sess.lead["consultation_mode"] = sess.lead.get("consultation_mode") or "experience_centre"
                booking = {"slot": slot, "mode": sess.lead["consultation_mode"]}
            else:
                log.warning("invalid slot id %r", data.get("slot_id"))

        reply = (data.get("reply") or "").strip() or "Sorry, could you say that again?"
        reply_lang = data.get("reply_language") or sess.language
        if reply_lang not in config.TTS_LANGUAGES:
            reply_lang = sess.language if sess.language in config.TTS_LANGUAGES else "en-IN"

        # Keep the model's own JSON in history so it stays consistent turn to turn.
        sess.history.append({"role": "assistant", "content": json.dumps(
            {"reply": reply, "reply_language": reply_lang, "extracted": updates,
             "action": action, "slot_id": data.get("slot_id")}, ensure_ascii=False)})
        sess.transcript.append({"role": "agent", "text": reply, "lang": reply_lang,
                                "t": datetime.now(IST).isoformat()})

        if action == "end_call":
            self.finish(sess)

        return {"reply": reply, "reply_language": reply_lang, "updates": updates,
                "action": action, "booking": booking}

    def finish(self, sess: Session) -> None:
        if sess.done:
            return
        sess.done = True
        sc = scoring.score_lead(sess.lead)
        sess.summary = self._summarise(sess, sc)
        sess.lead_id = store.save_lead({
            "session_id": sess.id, "started_at": sess.started_at,
            "ended_at": datetime.now(IST).isoformat(), "lead": sess.lead,
            "score": sc.to_dict(), "summary": sess.summary, "transcript": sess.transcript,
        })

    def _summarise(self, sess: Session, sc) -> str:
        convo = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in sess.transcript)
        prompt = [
            {"role": "system", "content": "You write crisp CRM notes for an interior-design sales team. English only."},
            {"role": "user", "content":
                f"Call transcript:\n{convo}\n\nExtracted lead: {json.dumps(sess.lead, ensure_ascii=False)}\n"
                f"Lead score: {sc.tier} ({sc.points}/100).\n\n"
                "Write a CRM note in at most 4 short lines: (1) who/what/where, (2) budget & timeline, "
                "(3) objections or special requests, (4) recommended next step for the designer. Plain text, no markdown."},
        ]
        try:
            return self.brain.chat(prompt, model=config.SUMMARY_MODEL, max_tokens=1500,
                                   temperature=0.2, reasoning="low").strip()
        except Exception as e:  # summary is nice-to-have; never break the call over it
            log.warning("summary failed: %s", e)
            return f"{sc.tier} lead ({sc.points}/100). {sc.next_action}"
