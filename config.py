"""Central configuration. Everything is overridable via environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")

# Mock mode lets you run the whole app without a key (scripted brain, no real STT/TTS).
MOCK_MODE = os.getenv("MOCK_MODE", "").lower() in {"1", "true", "yes"} or not SARVAM_API_KEY

# --- Models -----------------------------------------------------------------
# Conversation turns: the variant post-trained for real-time dialogue / voice agents.
CHAT_MODEL = os.getenv("CHAT_MODEL", "sarvam-105b-conversations")
# End-of-call CRM summary: the flagship reasoning model.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "sarvam-105b")
# Reasoning is ON by default on Sarvam-105B. For a voice agent we switch it off for latency.
# Set CHAT_REASONING=low|medium|high to turn it back on.
CHAT_REASONING = os.getenv("CHAT_REASONING", "") or None

STT_MODEL = os.getenv("STT_MODEL", "saaras:v4")        # saaras:v3 also works (no keyterms)
TTS_MODEL = os.getenv("TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "ritu")           # lowercase bulbul:v3 voice
TTS_PACE = float(os.getenv("TTS_PACE", "1.05"))
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "22050"))

# Words Saaras v4 should be biased towards (keyterm prompting).
STT_KEYTERMS = [
    "BHK", "modular kitchen", "wardrobe", "experience centre",
    "possession", "lakh", "villa", "false ceiling",
]

# Languages Bulbul v3 can speak. Anything else falls back to Indian English.
TTS_LANGUAGES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
                 "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}

# --- Business rules (edit these for the real deployment) -----------------------
AGENT_NAME = os.getenv("AGENT_NAME", "Riya")
BRAND = os.getenv("BRAND", "Acme Interiors")   # placeholder; set your client's name in .env

# Sample serviceable-city list for the prototype. Replace with the live list.
SERVICEABLE_CITIES = [
    "bengaluru", "bangalore", "mumbai", "navi mumbai", "thane", "pune", "hyderabad",
    "chennai", "delhi", "new delhi", "gurugram", "gurgaon", "noida", "ghaziabad",
    "kolkata", "mysuru", "mysore", "coimbatore", "visakhapatnam", "vizag", "lucknow",
    "kochi", "bhubaneswar", "nagpur", "ahmedabad", "surat", "jaipur", "chandigarh",
]

# The brand name itself is the most important keyterm for speech recognition.
STT_KEYTERMS = [BRAND] + STT_KEYTERMS

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LEADS_FILE = DATA_DIR / "leads.jsonl"
