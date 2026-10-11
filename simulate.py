"""Stress-test the agent with simulated customers before putting it on real calls.

Each persona is played by Sarvam-105B (live mode) or by a fixed script (mock mode). The
agent runs exactly as in production (text mode, no audio) and we check that the final
lead tier matches what a human sales manager would expect.

    python simulate.py            # all personas
    python simulate.py hot_hinglish cold_explorer
"""
import json
import sys
import time

from app import config, scoring
from app.agent import LeadAgent

PERSONAS = {
    "hot_hinglish": dict(
        expect="Hot", prefill={"name": "Priya", "city": "Bengaluru"}, lang="hi-IN",
        persona="Priya, 34, Bengaluru (Sarjapur Road). Speaks Hinglish. Just got keys to a 3 BHK apartment. "
                "Wants full home interiors, wants to start within 3-4 weeks, budget about 9 lakh, decides herself. "
                "Prefers visiting the experience centre, happy with the first slot offered.",
        script=["haan boliye", "3 BHK apartment hai Sarjapur Road pe", "keys mil gayi last week",
                "full home karwana hai", "within a month start karna hai", "around 9 lakh",
                "main khud decide karungi", "first one, experience centre", "thank you"]),
    "warm_family": dict(
        expect="Warm", prefill={"name": "Karthik", "city": "Chennai"}, lang="en-IN",
        persona="Karthik, Chennai (OMR). 2 BHK apartment, possession in about 4 months, under construction. "
                "Only kitchen and wardrobes. Budget around 2.5 lakh. Wants to start in 4-5 months. "
                "Must discuss with his wife before deciding. Polite but non-committal about booking; agrees to a video call slot.",
        script=["yes, tell me", "2 BHK apartment on OMR", "still under construction",
                "only kitchen and wardrobes", "maybe in 5 months", "around 2.5 lakh",
                "I need to discuss with my wife", "second one, video call", "ok bye"]),
    "cold_explorer": dict(
        expect="Cold", prefill={"name": "Rohit", "city": "Pune"}, lang="en-IN",
        persona="Rohit, Pune. Just browsing ideas, no fixed plan. 2 BHK flat, possession maybe next year, "
                "only wardrobes, budget not sure maybe 1 lakh. Not keen to book anything now.",
        script=["yeah ok", "2 BHK flat", "under construction", "just wardrobes", "just exploring",
                "maybe 1 lakh", "I will decide myself", "not now, maybe later. bye"]),
    "unserviceable": dict(
        expect="Disqualified", prefill={"name": "Ravi"}, lang="hi-IN",
        persona="Ravi, lives in Patna, Bihar. 3 BHK independent house, wants full home interiors soon.",
        script=["haan", "Patna mein ghar hai", "ok thank you"]),
    "not_interested": dict(
        expect="Disqualified", prefill={"name": "Sneha", "city": "Mumbai"}, lang="en-IN",
        persona="Sneha, Mumbai. Filled the form by mistake. Is annoyed and says she is not interested.",
        script=["who is this?", "I'm not interested, please don't call again"]),
}

CUSTOMER_PROMPT = """You are role-playing a customer receiving a phone call from a home-interiors company.
Persona: {persona}
Stay in character. Reply like a real person on a phone call: 1 short sentence, natural, in your persona's language style.
Only reveal information when asked. Output ONLY what you say, nothing else."""


def run(name, p, brain, voice):
    agent = LeadAgent(brain, voice)
    sess, turn = agent.start(p["prefill"])
    script = list(p["script"])
    convo = [{"role": "system", "content": CUSTOMER_PROMPT.format(persona=p["persona"])}]
    print(f"\n=== {name} ===\n  AGENT: {turn['reply']}")
    for _ in range(14):
        if sess.done:
            break
        if config.MOCK_MODE:
            if not script:
                break
            said = script.pop(0)
        else:
            # The customer sees the agent's line as the "user" side of its own chat.
            convo.append({"role": "user", "content": turn["reply"]})
            said = brain.chat(convo, model=config.SUMMARY_MODEL, max_tokens=600, temperature=0.7,
                              reasoning=None).strip().strip('"')
            convo.append({"role": "assistant", "content": said})
        print(f"  CUST : {said}")
        turn = agent.turn(sess.id, said, p["lang"])
        print(f"  AGENT: {turn['reply']}" + (f"   [{turn['action']}]" if turn["action"] != "none" else ""))
    if not sess.done:
        agent.finish(sess)
    sc = scoring.score_lead(sess.lead)
    return {"persona": name, "expected": p["expect"], "got": sc.tier, "points": sc.points,
            "turns": len([t for t in sess.transcript if t["role"] == "customer"]),
            "booked": sess.lead.get("consultation_slot") or "-", "pass": sc.tier == p["expect"]}


def main():
    if config.MOCK_MODE:
        from app.mock import MockBrain, MockVoice
        brain, voice = MockBrain(), MockVoice()
        print("MOCK MODE: scripted customers + rule-based agent. Set SARVAM_API_KEY for a real run.")
    else:
        from app.sarvam import SarvamClient
        brain = voice = SarvamClient()
    names = sys.argv[1:] or list(PERSONAS)
    t0, results = time.time(), [run(n, PERSONAS[n], brain, voice) for n in names]
    print("\n" + "-" * 86)
    print(f"{'persona':16}{'expected':14}{'got':14}{'pts':>5}{'turns':>7}  {'booked':22}result")
    for r in results:
        print(f"{r['persona']:16}{r['expected']:14}{r['got']:14}{r['points']:>5}{r['turns']:>7}  "
              f"{r['booked']:22}{'PASS' if r['pass'] else 'FAIL'}")
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} personas qualified correctly in {time.time() - t0:.0f}s")
    (config.DATA_DIR / "simulation_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
