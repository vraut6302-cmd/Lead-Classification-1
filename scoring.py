"""Deterministic, explainable lead scoring.

The LLM only *extracts* facts from the conversation. The score is computed here in plain
Python so the sales team can see exactly why a lead is Hot / Warm / Cold and so the
rules can be tuned without touching prompts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

# Allowed values for the categorical fields the LLM fills in.
ENUMS = {
    "property_type": ["apartment", "villa", "independent_house", "other"],
    "property_status": ["possession_received", "possession_within_3m",
                        "under_construction_later", "renovation"],
    "scope": ["full_home", "kitchen_wardrobes", "kitchen_only", "wardrobes_only",
              "specific_rooms"],
    "start_timeline": ["within_1m", "1_3m", "3_6m", "6m_plus", "exploring"],
    "decision_maker": ["self", "joint_present", "needs_consult"],
    "consultation_mode": ["experience_centre", "home_visit", "virtual"],
}

# Fields the agent must collect before it tries to book (in the order it should ask).
REQUIRED_FIELDS = ["city", "property_type", "bhk", "property_status", "scope",
                   "start_timeline", "budget_lakh", "decision_maker"]

# Rough minimum realistic budget (in lakh INR) per scope. Tune with real deal data.
MIN_BUDGET_LAKH = {
    ("full_home", 1): 3.0, ("full_home", 2): 4.5, ("full_home", 3): 6.5, ("full_home", 4): 9.0,
    "kitchen_wardrobes": 2.5, "kitchen_only": 1.5, "wardrobes_only": 1.0, "specific_rooms": 1.5,
}


def is_serviceable(city: str | None) -> bool | None:
    if not city:
        return None
    c = " " + re.sub(r"[^a-z ]", " ", city.lower()) + " "
    # Whole-word match, so "Whitefield, Bengaluru" matches but "Patna" doesn't match "Visakhapatnam".
    return any(f" {s} " in c for s in config.SERVICEABLE_CITIES)


def min_budget(scope: str | None, bhk: int | None) -> float | None:
    if not scope:
        return None
    if scope == "full_home":
        b = min(max(int(bhk or 2), 1), 4)
        return MIN_BUDGET_LAKH[("full_home", b)]
    return MIN_BUDGET_LAKH.get(scope)


@dataclass
class Score:
    points: int = 0
    tier: str = "Qualifying"        # Hot | Warm | Cold | Disqualified | Qualifying
    reasons: list[str] = field(default_factory=list)
    next_action: str = ""
    completeness: float = 0.0

    def to_dict(self) -> dict:
        return {"points": self.points, "tier": self.tier, "reasons": self.reasons,
                "next_action": self.next_action, "completeness": round(self.completeness, 2)}


def missing_fields(lead: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if lead.get(f) in (None, "", [])]


def score_lead(lead: dict, final: bool = True) -> Score:
    """final=False while the call is live: incomplete leads show as 'Qualifying'."""
    s = Score()
    filled = len(REQUIRED_FIELDS) - len(missing_fields(lead))
    s.completeness = filled / len(REQUIRED_FIELDS)

    if lead.get("not_interested"):
        s.tier, s.next_action = "Disqualified", "Close lead — customer not interested."
        s.reasons.append("Customer said they are not interested")
        return s

    def add(pts: int, why: str):
        s.points += pts
        s.reasons.append(f"{'+' if pts >= 0 else ''}{pts} {why}")

    # 1. Serviceability (gate)
    svc = is_serviceable(lead.get("city"))
    if svc is False:
        s.tier = "Disqualified"
        s.reasons.append(f"City '{lead.get('city')}' is outside the serviceable list")
        s.next_action = "Add to waitlist for city launch; send WhatsApp catalogue."
        return s
    if svc:
        add(15, "serviceable city")

    # 2. Timeline / urgency (strongest buying signal)
    tl = lead.get("start_timeline")
    add({"within_1m": 25, "1_3m": 18, "3_6m": 8, "6m_plus": 3, "exploring": 1}.get(tl, 0),
        f"start timeline: {tl or 'unknown'}")

    # 3. Budget fit vs. scope
    budget, need = lead.get("budget_lakh"), min_budget(lead.get("scope"), lead.get("bhk"))
    if budget and need:
        ratio = float(budget) / need
        if ratio >= 1:
            add(25, f"budget ₹{budget}L covers est. minimum ₹{need}L")
        elif ratio >= 0.7:
            add(14, f"budget ₹{budget}L slightly below est. ₹{need}L")
        else:
            add(4, f"budget ₹{budget}L well below est. ₹{need}L")
    elif budget:
        add(10, f"budget ₹{budget}L (scope unknown)")
    else:
        add(0, "budget not shared")

    # 4. Scope size
    add({"full_home": 15, "kitchen_wardrobes": 11, "kitchen_only": 7, "wardrobes_only": 5,
         "specific_rooms": 6}.get(lead.get("scope"), 0), f"scope: {lead.get('scope') or 'unknown'}")

    # 5. Property readiness
    add({"possession_received": 10, "possession_within_3m": 8, "renovation": 8,
         "under_construction_later": 3}.get(lead.get("property_status"), 0),
        f"property status: {lead.get('property_status') or 'unknown'}")

    # 6. Decision maker
    add({"self": 10, "joint_present": 10, "needs_consult": 5}.get(lead.get("decision_maker"), 0),
        f"decision maker: {lead.get('decision_maker') or 'unknown'}")

    # 7. Commitment signal
    if lead.get("consultation_slot"):
        add(5, "design consultation booked")

    s.points = min(s.points, 100)
    # Urgency caps: no near-term project means no hot lead, however good the rest looks.
    if tl == "exploring" and s.points >= 45:
        s.points = 44
        s.reasons.append("capped at Cold: customer is only exploring")
    elif tl in ("3_6m", "6m_plus") and s.points >= 70:
        s.points = 69
        s.reasons.append("capped at Warm: work starts 3+ months out")
    if s.points >= 70:
        s.tier = "Hot"
        s.next_action = ("Assign senior designer; confirm consultation on WhatsApp within 1 hour."
                         if lead.get("consultation_slot") else
                         "Call back within 1 hour to lock a consultation slot.")
    elif s.points >= 45:
        s.tier = "Warm"
        s.next_action = "Nurture: send design ideas + offers on WhatsApp, follow up in 7 days."
    else:
        s.tier = "Cold"
        s.next_action = "Long-term nurture drip; re-qualify in 60–90 days."
    if s.completeness < 0.75:
        if not final:
            s.tier = "Qualifying"
            s.next_action = "Agent is still collecting details."
        else:
            s.reasons.append(f"only {round(s.completeness * 100)}% of details collected — verify on callback")
    return s
