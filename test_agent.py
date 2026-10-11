import json
import os

os.environ["MOCK_MODE"] = "1"

import httpx
import pytest

from app import config, scoring
from app.agent import LeadAgent, normalise, parse_model_json
from app.mock import MockBrain, MockVoice
from app.sarvam import SarvamClient


@pytest.fixture(autouse=True)
def tmp_leads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEADS_FILE", tmp_path / "leads.jsonl")


# ------------------------------------------------------------------ scoring
def full_lead(**kw):
    base = dict(city="Pune", property_type="apartment", bhk=3, property_status="possession_received",
                scope="full_home", start_timeline="within_1m", budget_lakh=8, decision_maker="self")
    base.update(kw)
    return base


def test_hot_lead():
    s = scoring.score_lead(full_lead())
    assert s.tier == "Hot" and s.points >= 70


def test_cold_lead():
    s = scoring.score_lead(full_lead(start_timeline="exploring", budget_lakh=2, scope="wardrobes_only",
                                     property_status="under_construction_later", decision_maker="needs_consult"))
    assert s.tier == "Cold"


def test_unserviceable_city_disqualifies_without_substring_false_positive():
    assert scoring.is_serviceable("Patna") is False          # not matched by "Visakhapatnam"
    assert scoring.is_serviceable("Whitefield, Bengaluru") is True
    assert scoring.score_lead(full_lead(city="Patna")).tier == "Disqualified"


def test_not_interested():
    assert scoring.score_lead({"not_interested": True}).tier == "Disqualified"


# ------------------------------------------------------------------ parsing
def test_parse_fenced_json_and_plain_text():
    assert parse_model_json('```json\n{"reply":"hi","action":"none"}\n```')["reply"] == "hi"
    assert parse_model_json("Sure! Which city?")["reply"] == "Sure! Which city?"


def test_normalise():
    out = normalise({"bhk": "3 BHK", "budget_lakh": "800000", "scope": "Full Home",
                     "start_timeline": "next week", "consultation_mode": "experience center", "bogus": 1})
    assert out == {"bhk": 3, "budget_lakh": 8.0, "scope": "full_home", "consultation_mode": "experience_centre"}


# ------------------------------------------------------------------ full call (mock brain)
def test_mock_call_end_to_end():
    a = LeadAgent(MockBrain(), MockVoice())
    s, first = a.start({"name": "Ananya", "city": "Bengaluru"})
    assert config.BRAND in first["reply"]
    for u in ["yes", "3 BHK apartment", "got the keys", "full home", "within a month", "8 lakh",
              "myself", "first one", "thanks"]:
        a.turn(s.id, u, "en-IN")
    assert s.done and s.lead_id
    assert s.lead["consultation_slot"] == s.slots[0]["label"]
    assert scoring.score_lead(s.lead).tier == "Hot"
    saved = [json.loads(l) for l in config.LEADS_FILE.read_text().splitlines()]
    assert saved[0]["lead_id"] == s.lead_id


class ScriptedBrain:
    """Replays canned model outputs to test the agent's handling of the JSON protocol."""
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return self.outputs.pop(0) if self.outputs else "Summary."


def test_booking_and_invalid_values():
    brain = ScriptedBrain([
        json.dumps({"reply": "Namaste!", "reply_language": "hi-IN", "extracted": {}, "action": "none"}),
        json.dumps({"reply": "Ho gaya", "reply_language": "hi-IN",
                    "extracted": {"scope": "whole_house", "city": "Noida", "consultation_mode": "virtual"},
                    "action": "book_slot", "slot_id": "s2"}),
    ])
    a = LeadAgent(brain, MockVoice())
    s, _ = a.start({})
    r = a.turn(s.id, "S2 video call theek hai", "hi-IN")
    assert "scope" not in r["updates"]                      # invalid enum dropped
    assert s.lead["city"] == "Noida"
    assert s.lead["consultation_slot"] == s.slots[1]["label"]
    assert r["reply_language"] == "hi-IN"
    # The turn call disables reasoning and asks for JSON
    assert brain.calls[1][1]["json_mode"] is True and brain.calls[1][1]["reasoning"] is None
    assert "[lang: hi-IN]" in brain.calls[1][0][-1]["content"]


# ------------------------------------------------------------------ HTTP client request shapes
def make_client(handler):
    c = SarvamClient(api_key="sk_test", base_url="https://api.sarvam.ai")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_stt_request():
    seen = {}

    def h(req):
        seen["path"], seen["key"], seen["body"] = req.url.path, req.headers["api-subscription-key"], req.content
        return httpx.Response(200, json={"transcript": " namaste ", "language_code": "hi-IN"})

    out = make_client(h).transcribe(b"RIFF" + b"\0" * 3000)
    assert out == {"transcript": "namaste", "language_code": "hi-IN"}
    assert seen["path"] == "/speech-to-text" and seen["key"] == "sk_test"
    assert b"saaras:v4" in seen["body"] and config.BRAND.encode() in seen["body"]


def test_chat_degrades_when_optional_params_rejected():
    bodies = []

    def h(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(422, json={"error": {"message": "response_format unsupported"}})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "{}"}}]})

    out = make_client(h).chat([{"role": "user", "content": "hi"}], json_mode=True, reasoning=None)
    assert out == "{}"
    assert bodies[0]["model"] == "sarvam-105b-conversations"
    assert bodies[0]["reasoning_effort"] is None and "response_format" in bodies[0]
    assert "response_format" not in bodies[1]


def test_tts_request_and_language_fallback():
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"request_id": "x", "audios": ["UklGRg=="]})

    audio = make_client(h).speak("Hello", "as-IN")          # Assamese not in Bulbul v3 -> en-IN
    assert audio == "UklGRg=="
    assert seen["language_code"] == "en-IN" and seen["model"] == "bulbul:v3" and seen["speaker"] == "ritu"
