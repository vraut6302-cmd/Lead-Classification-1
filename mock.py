"""Offline stand-ins so the whole app runs without an API key (MOCK_MODE).

MockBrain is a keyword/regex rule engine that speaks the same JSON protocol as the real
Sarvam-105B prompt. It exists for demos without credits and for automated tests — the
real agent quality comes from Sarvam-105B.
"""
from __future__ import annotations

import json
import re

from . import config, scoring

QUESTIONS = {
    "city": "Which city is the home in, and which area?",
    "property_type": "Is it an apartment, a villa or an independent house?",
    "bhk": "How many bedrooms — 2 BHK, 3 BHK?",
    "property_status": "Have you received the keys yet, or is possession still due?",
    "scope": "Are you planning interiors for the full home, or just the kitchen and wardrobes?",
    "start_timeline": "When would you like the work to start?",
    "budget_lakh": "Roughly what budget do you have in mind for this?",
    "decision_maker": "Will you be deciding on this yourself, or along with family?",
}


def _extract(text: str) -> dict:
    t = text.lower()
    out: dict = {}
    if re.search(r"not interested|don'?t call|nahi chahiye|no thanks|stop calling", t):
        out["not_interested"] = True
        return out
    for city in ["bengaluru", "bangalore", "mumbai", "pune", "hyderabad", "chennai", "delhi",
                 "gurgaon", "gurugram", "noida", "kolkata", "patna", "ranchi", "indore",
                 "guwahati", "kanpur"]:
        if city in t:
            out["city"] = city.title()
            break
    if m := re.search(r"(\d)\s*-?\s*bhk", t):
        out["bhk"] = int(m.group(1))
    if "villa" in t:
        out["property_type"] = "villa"
    elif re.search(r"apartment|flat|society", t):
        out["property_type"] = "apartment"
    elif re.search(r"independent|bungalow|own house", t):
        out["property_type"] = "independent_house"
    if re.search(r"renovat|old house|redo", t):
        out["property_status"] = "renovation"
    elif re.search(r"got (the )?keys|keys (are )?(with me|received)|keys? mil gay|possession (is )?(done|received)|ready to move|already moved", t):
        out["property_status"] = "possession_received"
    elif re.search(r"possession.*(next month|soon|in \d (weeks|months))|keys.*next month", t):
        out["property_status"] = "possession_within_3m"
    elif "under construction" in t:
        out["property_status"] = "under_construction_later"
    if re.search(r"full home|entire|whole (home|house|flat)|complete home|everything", t):
        out["scope"] = "full_home"
    elif "kitchen" in t and "wardrobe" in t:
        out["scope"] = "kitchen_wardrobes"
    elif "kitchen" in t:
        out["scope"] = "kitchen_only"
    elif "wardrobe" in t:
        out["scope"] = "wardrobes_only"
    if re.search(r"immediately|asap|this month|right away|within a month|next week", t):
        out["start_timeline"] = "within_1m"
    elif re.search(r"(2|3|two|three|couple of) months", t):
        out["start_timeline"] = "1_3m"
    elif re.search(r"(4|5|6|six) months", t):
        out["start_timeline"] = "3_6m"
    elif re.search(r"next year|later|a year", t):
        out["start_timeline"] = "6m_plus"
    elif re.search(r"just (looking|exploring)|exploring|not sure", t):
        out["start_timeline"] = "exploring"
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|l\b)", t):
        out["budget_lakh"] = float(m.group(1))
    elif m := re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr\b)", t):
        out["budget_lakh"] = float(m.group(1)) * 100
    if re.search(r"together|both of us|with me on the call", t):
        out["decision_maker"] = "joint_present"
    elif re.search(r"wife|husband|family|parents|spouse|discuss", t):
        out["decision_maker"] = "needs_consult"
    elif re.search(r"myself|i decide|only me|my decision|i will decide|main khud|khud decide", t):
        out["decision_maker"] = "self"
    if re.search(r"experience (centre|center)|showroom|visit (the|your) (centre|center|store)", t):
        out["consultation_mode"] = "experience_centre"
    elif re.search(r"home visit|come (to my )?home|come over", t):
        out["consultation_mode"] = "home_visit"
    elif re.search(r"video|online|virtual|zoom|meet call", t):
        out["consultation_mode"] = "virtual"
    return out


class MockBrain:
    def chat(self, messages, model=None, **kw):
        system = messages[0]["content"]
        # Summary request (no JSON protocol)
        if "CRM note" in messages[-1]["content"]:
            return "Mock summary: lead details captured on call; see fields. Designer to review floor plan and share quote."

        lead = json.loads(re.search(r"CURRENT KNOWN LEAD DETAILS \(JSON\): (\{.*\})", system).group(1))
        slots = re.search(r"AVAILABLE SLOTS: (.*)", system).group(1).split("; ")
        last = messages[-1]["content"]
        prev_agent = next((json.loads(m["content"]).get("reply", "") for m in reversed(messages[:-1])
                           if m["role"] == "assistant"), "")

        def out(reply, extracted=None, action="none", slot_id=None):
            return json.dumps({"reply": reply, "reply_language": "en-IN",
                               "extracted": extracted or {}, "action": action, "slot_id": slot_id})

        if last.startswith("[call connected"):
            name = lead.get("name") or "there"
            return out(f"Hi {name}, this is {config.AGENT_NAME}, {config.BRAND}'s virtual assistant, calling about your "
                       "interiors enquiry. Is this a good time to talk for two minutes?")

        text = re.sub(r"^\[lang: [^\]]+\]\s*", "", last)
        ext = _extract(text)
        if ext.get("not_interested"):
            return out("No problem at all, sorry to disturb you. Have a lovely day!", ext, "end_call")

        # Keep the first answer for each field (a later "maybe later, bye" must not overwrite it).
        ext = {k: v for k, v in ext.items() if lead.get(k) in (None, "", False) or k == "consultation_mode"}
        merged = {**lead, **ext}
        if scoring.is_serviceable(merged.get("city")) is False:
            return out(f"Thank you! We're not in {merged['city']} yet, but I'll make sure you're "
                       "notified the moment we launch there. Have a great day!", ext, "end_call")

        if lead.get("consultation_slot"):
            return out("You're all set! Have a great day.", ext, "end_call")

        missing = [f for f in scoring.REQUIRED_FIELDS if merged.get(f) in (None, "")]
        if missing:
            return out(QUESTIONS[missing[0]], ext)

        offered = "Which works" in prev_agent
        if offered:
            pick = None
            if m := re.search(r"\bs(\d)\b", text.lower()):
                pick = f"S{m.group(1)}"
            elif re.search(r"first|1st|pehla", text.lower()):
                pick = slots[0].split("=")[0]
            elif re.search(r"second|2nd|dusra|later one", text.lower()):
                pick = slots[1].split("=")[0]
            if not pick and re.search(r"not now|later|bye|no thanks|abhi nahi", text.lower()):
                return out("No problem — I'll share some design ideas on WhatsApp and we can talk "
                           "whenever you're ready. Have a nice day!", ext, "end_call")
            if pick:
                label = next(s.split("=")[1] for s in slots if s.startswith(pick + "="))
                mode = merged.get("consultation_mode") or "experience_centre"
                ext.setdefault("consultation_mode", mode)
                return out(f"Done — your free design consultation is booked for {label}. "
                           f"A designer from {config.BRAND} will WhatsApp you shortly. Thank you!", ext,
                           "book_slot", pick)
        s1, s2 = slots[0].split("=")[1], slots[1].split("=")[1]
        return out(f"Great, a designer can help with that. Which works better — {s1} or {s2}? "
                   "And would you prefer our experience centre, a home visit, or a video call?", ext)


class MockVoice:
    """No real speech in mock mode: the browser uses its own speechSynthesis instead."""

    def transcribe(self, wav_bytes: bytes, language_code=None) -> dict:
        return {"transcript": "", "language_code": None}

    def speak(self, text: str, language_code: str = "en-IN") -> str:
        return ""
