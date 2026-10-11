"""Tiny JSONL 'CRM'. Swap save_lead() for a Salesforce / LeadSquared / HubSpot call in production."""
from __future__ import annotations

import csv
import io
import json
import uuid

from . import config


def save_lead(record: dict) -> str:
    record = {"lead_id": "LD-" + uuid.uuid4().hex[:6].upper(), **record}
    with open(config.LEADS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["lead_id"]


def list_leads() -> list[dict]:
    if not config.LEADS_FILE.exists():
        return []
    rows = []
    for line in config.LEADS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return list(reversed(rows))


CSV_COLUMNS = ["lead_id", "ended_at", "tier", "points", "name", "city", "locality",
               "property_type", "bhk", "property_status", "scope", "budget_lakh",
               "start_timeline", "decision_maker", "consultation_mode", "consultation_slot",
               "callback_time", "next_action", "summary"]


def leads_csv() -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in list_leads():
        w.writerow({**r.get("lead", {}), "lead_id": r.get("lead_id"), "ended_at": r.get("ended_at"),
                    "tier": r["score"]["tier"], "points": r["score"]["points"],
                    "next_action": r["score"]["next_action"], "summary": r.get("summary", "")})
    return buf.getvalue()
