"""
KeaBuilder Lead Intelligence System
Q1: AI-powered lead classification and intelligent response generation

Architecture:
  Form Submission → Pre-processing → AI Classification → Score Engine → Response Generator → CRM
"""

import json
import re
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class LeadInput:
    name: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    job_title: Optional[str]
    budget: Optional[str]
    timeline: Optional[str]
    message: Optional[str]
    source: Optional[str]       # e.g., landing_page, webinar, referral
    funnel_id: Optional[str]    # KeaBuilder funnel identifier
    country: Optional[str]
    website: Optional[str]


@dataclass
class ClassifiedLead:
    lead_id: str
    tier: str                   # hot | warm | cold
    score: int                  # 0-100
    confidence: float           # 0.0-1.0
    intent_signals: list
    missing_fields: list
    recommended_action: str
    personalized_response: str
    follow_up_sequence: str
    classified_at: str
    raw_input: dict


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """
You are LeadIQ, KeaBuilder's expert lead scoring AI with deep knowledge in B2B SaaS sales.

Your task is to analyze a form submission and classify the lead with extreme accuracy.

SCORING FRAMEWORK (0–100):
─────────────────────────────
Budget Signals          → 0–25 pts
  - Explicit budget stated + realistic     : 20–25
  - Vague budget ("some", "flexible")      : 10–15
  - No budget mentioned                    : 0–5

Timeline Urgency        → 0–25 pts
  - "ASAP", "this week/month"              : 20–25
  - "next quarter", "6 months"             : 10–15
  - "someday", "just exploring"            : 0–5

Authority/Role          → 0–25 pts
  - CEO, Founder, VP, Director, Owner      : 20–25
  - Manager, Team Lead                     : 10–15
  - Individual contributor / Unknown       : 0–5

Fit & Intent            → 0–25 pts
  - Detailed specific message, website URL : 20–25
  - Generic inquiry, referral              : 10–15
  - Spam-like, very vague                  : 0–5

TIERS:
  HOT  → Score 70–100 | Ready to buy, respond within 1 hour
  WARM → Score 40–69  | Interested but needs nurturing
  COLD → Score 0–39   | Early stage or poor fit

HANDLING INCOMPLETE INPUTS:
  - Missing fields → infer what you can, flag what's missing
  - Spam-like input → score 0–5, flag for review
  - Ambiguous message → ask clarifying question in response
  - No company = B2C or small business (adjust scoring)

OUTPUT FORMAT (strict JSON, no markdown):
{
  "tier": "hot|warm|cold",
  "score": <0-100>,
  "confidence": <0.0-1.0>,
  "intent_signals": ["signal1", "signal2"],
  "missing_fields": ["field1", "field2"],
  "score_breakdown": {
    "budget": <0-25>,
    "timeline": <0-25>,
    "authority": <0-25>,
    "fit_intent": <0-25>
  },
  "recommended_action": "string",
  "follow_up_sequence": "immediate_call|email_sequence_1|email_sequence_2|nurture_drip",
  "classification_reason": "2-3 sentence explanation"
}
"""

CLASSIFICATION_USER_PROMPT = """
Analyze this KeaBuilder lead submission and classify it:

FORM DATA:
Name: {name}
Email: {email}
Phone: {phone}
Company: {company}
Job Title: {job_title}
Budget: {budget}
Timeline: {timeline}
Message: {message}
Source: {source}
Country: {country}
Website: {website}
Funnel ID: {funnel_id}
Submitted At: {submitted_at}

Classify this lead now.
"""

RESPONSE_GENERATION_SYSTEM_PROMPT = """
You are the AI response writer for KeaBuilder, a SaaS platform for funnels and marketing automation.

Your job: Write personalized, human-sounding first responses to incoming leads.

VOICE & TONE RULES:
1. Sound like a real person (Sales Director / Account Executive)
2. Use their FIRST NAME naturally — once at the start, not repetitively
3. Reference SPECIFIC details from their form (company name, problem, timeline)
4. Never mention "AI", "automated", or "bot"
5. One concrete value proposition tied to their situation
6. Clear single CTA (call, demo, free trial link)
7. Keep it under 120 words
8. Natural sign-off with real human name: "Alex from KeaBuilder"

PERSONALIZATION TRIGGERS:
- If budget mentioned → acknowledge it, show ROI
- If specific timeline → create urgency around it
- If company mentioned → reference their likely industry challenges
- If referral source → mention it warmly
- If incomplete info → gently ask the one most important missing question

TONE BY TIER:
  HOT  → Enthusiastic, direct, immediate call-to-action
  WARM → Helpful, curious, offer free resource + soft CTA
  COLD → Educational, low-pressure, curiosity-driven

OUTPUT: Plain text email/message only. No subject line. No markdown.
"""

RESPONSE_GENERATION_USER_PROMPT = """
Write a personalized first-touch response for this lead:

LEAD TIER: {tier}
LEAD SCORE: {score}/100

CONTACT INFO:
Name: {name}
Company: {company}
Job Title: {job_title}

THEIR CONTEXT:
Budget: {budget}
Timeline: {timeline}
Message: {message}
Source: {source}
Missing Information: {missing_fields}

Intent Signals Detected: {intent_signals}
Recommended Action: {recommended_action}

Write the response now:
"""


# ─────────────────────────────────────────────
# PRE-PROCESSING LAYER
# ─────────────────────────────────────────────

def preprocess_lead(raw: dict) -> tuple[LeadInput, list[str]]:
    """
    Sanitize, normalize, and detect missing fields before AI processing.
    Returns (LeadInput, list_of_issues)
    """
    issues = []

    # Normalize name
    name = raw.get("name", "").strip().title()
    if not name:
        issues.append("missing_name")
        name = "Friend"

    # Email validation
    email = (raw.get("email") or "").strip().lower()
    if not email or not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        issues.append("invalid_email")

    # Disposable email check (simplified)
    disposable_domains = ["mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com"]
    if any(d in email for d in disposable_domains):
        issues.append("disposable_email")

    # Budget normalization
    budget_raw = (raw.get("budget") or "")
    budget = normalize_budget(budget_raw)

    # Phone normalization
    phone = (raw.get("phone") or "").strip()
    phone = re.sub(r'[^\d+\-\s()]', '', phone) if phone else None

    return LeadInput(
        name=name,
        email=email,
        phone=phone or None,
        company=(raw.get("company") or "").strip() or None,
        job_title=(raw.get("job_title") or "").strip() or None,
        budget=budget,
        timeline=(raw.get("timeline") or "").strip() or None,
        message=(raw.get("message") or "").strip() or None,
        source=raw.get("source") or "organic",
        funnel_id=raw.get("funnel_id") or None,
        country=raw.get("country") or None,
        website=(raw.get("website") or "").strip() or None,
    ), issues


def normalize_budget(budget_str: str) -> Optional[str]:
    """Normalize various budget formats to a standard representation."""
    if not budget_str:
        return None
    budget_str = budget_str.strip().lower()
    # Map common patterns
    budget_map = {
        "under 1k": "<$1,000/mo",
        "1k-5k": "$1,000–$5,000/mo",
        "5k-10k": "$5,000–$10,000/mo",
        "10k+": "$10,000+/mo",
        "enterprise": "Enterprise (custom)",
    }
    for key, val in budget_map.items():
        if key in budget_str:
            return val
    return budget_str  # Return as-is if no normalization found


# ─────────────────────────────────────────────
# RULE-BASED SCORING (runs BEFORE AI — faster & cheaper)
# ─────────────────────────────────────────────

HOT_TITLE_KEYWORDS = ["ceo", "cto", "coo", "cmo", "founder", "co-founder", "owner", "vp", "vice president", "director", "president", "partner"]
WARM_TITLE_KEYWORDS = ["manager", "lead", "head of", "supervisor", "coordinator", "specialist"]
URGENT_TIMELINE_KEYWORDS = ["asap", "immediately", "urgent", "this week", "this month", "right away", "now"]
DEMO_INTENT_KEYWORDS = ["demo", "trial", "pricing", "quote", "proposal", "schedule", "book", "call", "meeting", "interested in buying"]

def rule_based_prescore(lead: LeadInput) -> dict:
    """
    Fast rule-based pre-scoring to set baseline before AI classification.
    This saves API calls for obvious cases.
    """
    score = 0
    signals = []

    title = (lead.job_title or "").lower()
    message = (lead.message or "").lower()
    timeline = (lead.timeline or "").lower()

    # Authority check
    if any(kw in title for kw in HOT_TITLE_KEYWORDS):
        score += 22
        signals.append("decision_maker_title")
    elif any(kw in title for kw in WARM_TITLE_KEYWORDS):
        score += 12
        signals.append("influencer_title")

    # Timeline urgency
    if any(kw in timeline for kw in URGENT_TIMELINE_KEYWORDS):
        score += 20
        signals.append("urgent_timeline")
    elif timeline and "quarter" in timeline:
        score += 10
        signals.append("near_term_timeline")

    # Intent signals in message
    if any(kw in message for kw in DEMO_INTENT_KEYWORDS):
        score += 18
        signals.append("explicit_demo_intent")

    # Budget provided
    if lead.budget:
        score += 15
        signals.append("budget_disclosed")

    # Company provided (B2B signal)
    if lead.company:
        score += 8
        signals.append("company_provided")

    # Phone provided (high intent)
    if lead.phone:
        score += 5
        signals.append("phone_provided")

    # Website provided
    if lead.website:
        score += 5
        signals.append("website_provided")

    # Referral source boost
    if lead.source in ["referral", "partner", "word_of_mouth"]:
        score += 10
        signals.append("referral_source")

    return {"prescore": min(score, 100), "signals": signals}


# ─────────────────────────────────────────────
# MAIN CLASSIFIER (simulated — in prod, calls OpenAI/Claude API)
# ─────────────────────────────────────────────

def classify_and_respond(raw_form_data: dict) -> ClassifiedLead:
    """
    Full pipeline:
    1. Preprocess & validate
    2. Rule-based pre-scoring
    3. AI classification (simulated)
    4. AI response generation (simulated)
    5. Return structured output
    """
    import uuid

    lead, issues = preprocess_lead(raw_form_data)
    prescore_result = rule_based_prescore(lead)

    # ── In production, this calls the AI API ──────────────────────
    # classification = call_ai_api(
    #     system=CLASSIFICATION_SYSTEM_PROMPT,
    #     user=CLASSIFICATION_USER_PROMPT.format(**asdict(lead), submitted_at=datetime.utcnow().isoformat())
    # )
    # response_text = call_ai_api(
    #     system=RESPONSE_GENERATION_SYSTEM_PROMPT,
    #     user=RESPONSE_GENERATION_USER_PROMPT.format(...)
    # )
    # ─────────────────────────────────────────────────────────────

    # Simulated AI output (deterministic for demo):
    prescore = prescore_result["prescore"]
    signals = prescore_result["signals"]

    tier = "hot" if prescore >= 70 else ("warm" if prescore >= 40 else "cold")
    confidence = min(0.95, 0.5 + prescore / 200)

    # Determine missing fields
    missing = []
    if not lead.budget:      missing.append("budget")
    if not lead.timeline:    missing.append("timeline")
    if not lead.job_title:   missing.append("job_title")
    if not lead.phone:       missing.append("phone")
    if not lead.company:     missing.append("company")

    # Generate simulated personalized response
    first_name = lead.name.split()[0]
    company_str = f" at {lead.company}" if lead.company else ""
    message_ref = ""
    if lead.message and len(lead.message) > 20:
        message_ref = f" Your note about '{lead.message[:40]}...' really stood out."

    if tier == "hot":
        personalized_response = (
            f"Hi {first_name}, thanks for reaching out{company_str}!{message_ref} "
            f"With your timeline in mind, I'd love to jump on a quick 20-minute call this week to show you "
            f"exactly how KeaBuilder can plug into your workflow. "
            f"Does Thursday or Friday work? Here's my calendar: [calendar_link]\n\n"
            f"— Alex from KeaBuilder"
        )
        action = "Assign to senior AE, call within 1 hour"
        sequence = "immediate_call"
    elif tier == "warm":
        personalized_response = (
            f"Hi {first_name}, great to connect{company_str}!{message_ref} "
            f"I'd love to share a quick case study of how similar businesses are using KeaBuilder "
            f"to automate their funnels and save 10+ hours/week. "
            f"Would it make sense to schedule a short demo? Happy to work around your calendar.\n\n"
            f"— Alex from KeaBuilder"
        )
        action = "Send nurture email + schedule discovery call within 24 hours"
        sequence = "email_sequence_1"
    else:
        budget_note = ""
        if "budget" in missing:
            budget_note = " Quick question — do you have a rough monthly budget in mind? That helps me point you in the right direction."
        personalized_response = (
            f"Hi {first_name}, thanks for your interest in KeaBuilder!{message_ref} "
            f"We help businesses like yours build high-converting funnels without a developer.{budget_note} "
            f"I'll send over our getting-started guide in case you want to explore at your own pace. "
            f"Feel free to book a call anytime: [calendar_link]\n\n"
            f"— Alex from KeaBuilder"
        )
        action = "Add to cold nurture drip (7-email sequence)"
        sequence = "nurture_drip"

    return ClassifiedLead(
        lead_id=str(uuid.uuid4()),
        tier=tier,
        score=prescore,
        confidence=round(confidence, 2),
        intent_signals=signals,
        missing_fields=missing,
        recommended_action=action,
        personalized_response=personalized_response,
        follow_up_sequence=sequence,
        classified_at=datetime.utcnow().isoformat() + "Z",
        raw_input=raw_form_data,
    )


# ─────────────────────────────────────────────
# DEMO: SAMPLE INPUTS & OUTPUTS
# ─────────────────────────────────────────────

SAMPLE_INPUTS = {
    "hot_lead": {
        "name": "Rajiv Mehta",
        "email": "rajiv@growthstacks.io",
        "phone": "+91-98765-43210",
        "company": "GrowthStacks",
        "job_title": "Founder & CEO",
        "budget": "5k-10k",
        "timeline": "ASAP — trying to launch next month",
        "message": "We're scaling our agency funnels and need a proper automation stack. Currently losing leads through the cracks. Ready to demo this week.",
        "source": "referral",
        "funnel_id": "fkb_004_agency_lp",
        "country": "India",
        "website": "growthstacks.io",
    },
    "warm_lead": {
        "name": "Priya Sharma",
        "email": "priya.sharma@brandnova.com",
        "phone": None,
        "company": "BrandNova",
        "job_title": "Marketing Manager",
        "budget": None,
        "timeline": "Next quarter",
        "message": "Exploring options for our lead generation funnel. Have seen KeaBuilder mentioned in a few communities.",
        "source": "organic_search",
        "funnel_id": "fkb_001_main",
        "country": "India",
        "website": None,
    },
    "cold_lead": {
        "name": "user123",
        "email": "test@gmail.com",
        "phone": None,
        "company": None,
        "job_title": None,
        "budget": None,
        "timeline": None,
        "message": "hi",
        "source": "unknown",
        "funnel_id": None,
        "country": None,
        "website": None,
    },
    "incomplete_lead": {
        "name": "Ankit Verma",
        "email": "ankit@techstartup.in",
        "phone": "+91-77889-00112",
        "company": "TechStartup",
        "job_title": None,           # Missing
        "budget": None,              # Missing
        "timeline": "Soon",          # Vague
        "message": "Interested in the platform for managing our sales team's follow-ups.",
        "source": "webinar",
        "funnel_id": "fkb_002_webinar",
        "country": "India",
        "website": "techstartup.in",
    }
}


if __name__ == "__main__":
    print("=" * 70)
    print("KeaBuilder Lead Intelligence System — Demo Run")
    print("=" * 70)

    for label, raw_input in SAMPLE_INPUTS.items():
        result = classify_and_respond(raw_input)
        print(f"\n{'─'*70}")
        print(f"📋 SCENARIO: {label.upper().replace('_', ' ')}")
        print(f"{'─'*70}")
        print(json.dumps(asdict(result), indent=2))

    print("\n" + "=" * 70)
    print("Demo complete.")
