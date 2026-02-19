"""
chatbot_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Conversational churn prediction pipeline.

Flow:
  User message
    → LLM extracts / asks for missing customer features (multi-turn)
    → Once all features are collected, build encoded feature vector
    → Run RandomForest churn model
    → LLM formats result as plain-English explanation for marketing team

LLM backend: Groq Cloud (open-source models, cloud-hosted, free tier)
  Models   : llama-3.3-70b-versatile, llama3-8b-8192, mixtral-8x7b-32768 …
  Get key  : https://console.groq.com → API Keys → Create key (no credit card)
  Set env  : export GROQ_API_KEY="gsk_..."     (Windows: set GROQ_API_KEY=...)
  Docs     : https://console.groq.com/docs/openai
────────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import re
import joblib
import numpy as np
from groq import Groq
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # loads .env file into os.environ before anything else runs

# ── Config ────────────────────────────────────────────────────────────────────
# Model options (all open-source, all free on Groq's free tier):
#   "llama-3.3-70b-versatile"  ← best quality,  ~300 tok/s
#   "llama3-8b-8192"           ← fastest,        ~800 tok/s
#   "mixtral-8x7b-32768"       ← long context,   good for long multi-turn chats
GROQ_MODEL = "llama-3.3-70b-versatile"
MODEL_PATH = "assets/model/churn_model.pkl"
FEATURES_PATH = "assets/model/feature_names.json"

# Groq client — reads GROQ_API_KEY from environment variable automatically
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Load ML artifacts once at import time ─────────────────────────────────────
_model = joblib.load(MODEL_PATH)
_feature_names: list[str] = json.load(open(FEATURES_PATH))


# ── Feature schema ─────────────────────────────────────────────────────────────
# Describes every input the model needs.
# Used both as an extraction target for the LLM and as validation.
FEATURE_SCHEMA = {
    # numeric
    "senior_citizen":   {"type": "int",   "values": [0, 1]},
    "tenure":           {"type": "float", "min": 0,   "max": 72},
    "monthly_charges":  {"type": "float", "min": 0},
    "total_charges":    {"type": "float", "min": 0},
    # binary encoded (0/1)
    "gender":           {"type": "binary", "map": {"male": 1, "female": 0}},
    "is_married":       {"type": "binary", "map": {"yes": 1, "no": 0}},
    "dependents":       {"type": "binary", "map": {"yes": 1, "no": 0}},
    "phone_service":    {"type": "binary", "map": {"yes": 1, "no": 0}},
    "paperless_billing": {"type": "binary", "map": {"yes": 1, "no": 0}},
    # ordinal encoded (0/1/2)
    "dual":             {"type": "ordinal", "map": {"no_phone_service": 0, "no": 1, "yes": 2}},
    "online_security":  {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "online_backup":    {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "device_protection": {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "tech_support":     {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "streaming_tv":     {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "streaming_movies": {"type": "ordinal", "map": {"no_internet_service": 0, "no": 1, "yes": 2}},
    "contract":         {"type": "ordinal", "map": {"month_to_month": 0, "one_year": 1, "two_year": 2}},
    # one-hot encoded (internet_service)
    "internet_service_dsl":        {"type": "ohe_member", "group": "internet_service"},
    "internet_service_fiber_optic": {"type": "ohe_member", "group": "internet_service"},
    "internet_service_no":         {"type": "ohe_member", "group": "internet_service"},
    # one-hot encoded (payment_method)
    "payment_method_bank_transfer_automatic": {"type": "ohe_member", "group": "payment_method"},
    "payment_method_credit_card_automatic":   {"type": "ohe_member", "group": "payment_method"},
    "payment_method_electronic_check":        {"type": "ohe_member", "group": "payment_method"},
    "payment_method_mailed_check":            {"type": "ohe_member", "group": "payment_method"},
}

# Human-friendly field names the LLM should ask the user for
HUMAN_FIELDS = {
    "gender":            ("Gender",             "male or female"),
    "senior_citizen":    ("Senior citizen?",    "yes or no"),
    "is_married":        ("Married?",           "yes or no"),
    "dependents":        ("Has dependents?",    "yes or no"),
    "tenure":            ("Tenure",             "number of months with the company"),
    "phone_service":     ("Phone service?",     "yes or no"),
    "dual":              ("Second phone line (dual)?", "yes / no / no_phone_service"),
    "internet_service":  ("Internet service",   "dsl / fiber_optic / no"),
    "online_security":   ("Online security?",   "yes / no / no_internet_service"),
    "online_backup":     ("Online backup?",     "yes / no / no_internet_service"),
    "device_protection": ("Device protection plan for internet devices?", "yes / no / no_internet_service"),
    "tech_support":      ("Tech support?",      "yes / no / no_internet_service"),
    "streaming_tv":      ("Streaming TV?",      "yes / no / no_internet_service"),
    "streaming_movies":  ("Streaming movies?",  "yes / no / no_internet_service"),
    "contract":          ("Contract type",      "month_to_month / one_year / two_year"),
    "paperless_billing": ("Paperless billing?", "yes or no"),
    "payment_method":    ("Payment method",     "electronic_check / mailed_check / bank_transfer_automatic / credit_card_automatic"),
    "monthly_charges":   ("Monthly charges",    "amount in $"),
    "total_charges":     ("Total charges",      "total amount billed so far in $"),
}

# Groups that map to OHE columns
OHE_GROUPS = {
    "internet_service": {
        "dsl":         "internet_service_dsl",
        "fiber_optic": "internet_service_fiber_optic",
        "no":          "internet_service_no",
    },
    "payment_method": {
        "bank_transfer_automatic":  "payment_method_bank_transfer_automatic",
        "credit_card_automatic":    "payment_method_credit_card_automatic",
        "electronic_check":         "payment_method_electronic_check",
        "mailed_check":             "payment_method_mailed_check",
    },
}

# ── LLM helper ────────────────────────────────────────────────────────────────


def _call_groq(messages: list[dict], temperature: float = 0.3) -> str:
    """
    Send messages to Groq Cloud and return the assistant reply.
    The Groq SDK is OpenAI-compatible — same interface, much faster inference.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get your free key at https://console.groq.com\n"
            "Then run:  export GROQ_API_KEY='gsk_...'"
        )
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Feature extraction ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a strict data extraction engine for a telecom churn prediction system.

Your ONLY output must be a single valid JSON object. No prose. No markdown. No explanation. No code fences.

EXTRACTION RULES:
- Extract every feature that can be inferred from the conversation, even indirectly.
- Normalize all values: lowercase, underscores instead of spaces/hyphens.
- Never include a field if you are genuinely unsure — omit it rather than guess.
- "pays by electronic check" → payment_method = "electronic_check" AND paperless_billing = "yes"
- "senior citizen" or age >= 65 → senior_citizen = 1 ; age clearly under 65 → senior_citizen = 0
- "married" → is_married = "yes" ; "single/unmarried" → is_married = "no"
- "has dependents/children" → dependents = "yes"
- "fiber optic" or "fiber" → internet_service = "fiber_optic"
- "no internet" → internet_service = "no" AND set online_security/online_backup/device_protection/tech_support/streaming_tv/streaming_movies = "no_internet_service"
- "no phone service" → phone_service = "no" AND dual = "no_phone_service"
- "streams TV and movies" → streaming_tv = "yes", streaming_movies = "yes"
- "month-to-month" or "monthly contract" → contract = "month_to_month"
- "one year contract" → contract = "one_year" ; "two year" → contract = "two_year"
- "electronic check" → payment_method = "electronic_check"
- "mailed check" or "postal check" → payment_method = "mailed_check"
- "bank transfer" → payment_method = "bank_transfer_automatic"
- "credit card" → payment_method = "credit_card_automatic"
- "$X/month" or "X dollars monthly" → monthly_charges = X (number only)
- "total of $X" or "billed X so far" → total_charges = X (number only)
- "X months with us" or "tenure of X" → tenure = X (number only)
- "no online security" → online_security = "no"
- "no backup" or "no online backup" → online_backup = "no"
- "no tech support" → tech_support = "no"
- "no device protection" → device_protection = "no"
- "paperless" or "e-billing" → paperless_billing = "yes" ; "paper bill" → paperless_billing = "no"
- "dual line" or "two lines" → dual = "yes" ; "single line" or "one line" → dual = "no"
- If phone_service = "no" is confirmed → dual = "no_phone_service" (infer automatically)
- "no second line" or "just one line" → dual = "no"

VALID FIELD VALUES:
gender: male|female | senior_citizen: 0|1 | is_married: yes|no | dependents: yes|no
tenure: <number> | phone_service: yes|no | dual: yes|no|no_phone_service
internet_service: dsl|fiber_optic|no
online_security/online_backup/device_protection/tech_support/streaming_tv/streaming_movies: yes|no|no_internet_service
contract: month_to_month|one_year|two_year | paperless_billing: yes|no
payment_method: electronic_check|mailed_check|bank_transfer_automatic|credit_card_automatic
monthly_charges: <number> | total_charges: <number>

EXAMPLE:
Input: "Male, 58yo senior, married, dependents, 12 months, phone, fiber optic, no security/backup/tech support, streams TV+movies, month-to-month, electronic check, $95/month, total $1140"
Output: {"gender":"male","senior_citizen":1,"is_married":"yes","dependents":"yes","tenure":12,"phone_service":"yes","internet_service":"fiber_optic","online_security":"no","online_backup":"no","tech_support":"no","streaming_tv":"yes","streaming_movies":"yes","contract":"month_to_month","payment_method":"electronic_check","paperless_billing":"yes","monthly_charges":95,"total_charges":1140}

Output JSON ONLY. Nothing else."""


def extract_features_from_conversation(history: list[dict]) -> dict:
    """Ask the LLM to parse collected features from the full conversation."""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": "Extract ALL customer features mentioned anywhere in this conversation. Be thorough — use the alias rules. Return JSON only, no other text."}
    ]
    # zero temp = deterministic extraction
    raw = _call_groq(messages, temperature=0.0)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    # Sometimes model wraps in extra text — grab the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ── Feature encoding ──────────────────────────────────────────────────────────

def encode_collected_features(collected: dict) -> Optional[np.ndarray]:
    """
    Convert the raw collected dict into the encoded feature vector
    that the model expects (same order as feature_names.json).

    Returns None if any required feature is still missing.
    """
    vector = {}

    for fname, schema in FEATURE_SCHEMA.items():
        ftype = schema["type"]

        if ftype in ("int", "float"):
            # Direct numeric fields
            raw_key = fname  # e.g. "tenure", "monthly_charges"
            if raw_key not in collected:
                return None
            vector[fname] = float(collected[raw_key])

        elif ftype == "binary":
            if fname not in collected:
                return None
            val = str(collected[fname]).lower()
            if val not in schema["map"]:
                return None
            vector[fname] = schema["map"][val]

        elif ftype == "ordinal":
            if fname not in collected:
                return None
            val = str(collected[fname]).lower()
            if val not in schema["map"]:
                return None
            vector[fname] = schema["map"][val]

        elif ftype == "ohe_member":
            group = schema["group"]
            if group not in collected:
                return None
            chosen = str(collected[group]).lower()
            col_for_chosen = OHE_GROUPS[group].get(chosen)
            if col_for_chosen is None:
                return None
            # Set 1 for chosen category, 0 for others
            for cat_val, col_name in OHE_GROUPS[group].items():
                vector[col_name] = 1 if col_name == col_for_chosen else 0

    # Build ordered array
    try:
        arr = np.array([[vector[f] for f in _feature_names]], dtype=float)
        return arr
    except KeyError:
        return None


def get_missing_fields(collected: dict) -> list[str]:
    """Return list of human-friendly field names still needed."""
    needed_raw_keys = set()
    for fname, schema in FEATURE_SCHEMA.items():
        ftype = schema["type"]
        if ftype in ("int", "float", "binary", "ordinal"):
            needed_raw_keys.add(fname)
        elif ftype == "ohe_member":
            # need group key, not column name
            needed_raw_keys.add(schema["group"])

    missing = []
    for raw_key in needed_raw_keys:
        if raw_key not in collected:
            label, hint = HUMAN_FIELDS.get(raw_key, (raw_key, ""))
            missing.append(f"{label} ({hint})")

    return missing


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_churn(feature_vector: np.ndarray) -> dict:
    """Run model and return churn probability + risk tier."""
    proba = _model.predict_proba(feature_vector)[0][1]
    if proba >= 0.70:
        risk = "HIGH"
    elif proba >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {"churn_probability": round(float(proba), 4), "risk_tier": risk}


# ── Result explanation ────────────────────────────────────────────────────────

EXPLANATION_SYSTEM_PROMPT = """You are a churn prediction assistant reporting results to a marketing agent.
You just ran the churn model on THEIR CUSTOMER. Report the result, then stop.

STRICT RULES:
- Speak to the AGENT about THEIR CUSTOMER — always third person ("this customer", "he", "she").
- 3 sentences maximum: (1) state churn probability + risk tier, (2) name the 1-2 biggest risk factors from the profile, (3) one specific action the agent should take.
- Do NOT say "I" or "we" ran a model — just state the result.
- Do NOT invite further conversation, do NOT summarize again, do NOT ask follow-up questions.
- Do NOT use technical ML terms.

GOOD example:
  "This customer has an 85.8% chance of churning — HIGH risk. The biggest drivers are his month-to-month contract and fiber optic plan with no add-ons. Recommend calling him within 48 hours to offer a one-year contract with a bundled security package."

BAD example:
  "Before we move on, let me summarize..." ← never recap after result is given
  "I ran the model and found..." ← don't mention the model
"""


def explain_result(prediction: dict, collected_features: dict, history: list[dict]) -> str:
    """Ask the LLM to generate a human-friendly explanation of the prediction."""
    summary = (
        f"Churn probability: {prediction['churn_probability']*100:.1f}% "
        f"(Risk tier: {prediction['risk_tier']})\n"
        f"Customer profile: {json.dumps(collected_features, indent=2)}"
    )
    messages = [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user",   "content": summary},
    ]
    return _call_groq(messages)


# ── Conversation orchestrator ─────────────────────────────────────────────────

GREETING = (
    "Hello! I'm your churn prediction assistant. "
    "I'll help you assess whether a customer is at risk of churning. "
    "Please tell me about the customer — you can share details naturally "
    "and I'll ask follow-up questions for anything I still need."
)

COLLECTION_SYSTEM_PROMPT = """You are a churn prediction assistant helping a marketing agent gather information about THEIR CUSTOMER.

You are talking to the MARKETING AGENT, not the customer. Always refer to the customer in third person ("the customer", "he", "she", "they").

STRICT RULES:
- NEVER ask for fields listed under "Already collected" — those are confirmed and final.
- Ask for at most 2 missing fields per message, phrased as simple yes/no or multiple-choice questions.
- Be concise — one short sentence per question max.
- Do not use technical terms. Use plain business language.
- Do not say "I'm trying to understand your situation" — you are asking about the CUSTOMER.

GOOD examples:
  "Does the customer have device protection? Also, does he have a second phone line?"
  "Two quick questions: Is the customer on paperless billing? And does she have device protection?"

BAD examples (never do these):
  "I'm trying to understand your situation..."  ← wrong, agent is not the customer
  "Does the customer have online security?"     ← already collected, never re-ask
"""


class ChurnChatSession:
    """
    Stateful conversation session. One instance per user/conversation.
    Holds message history and collected features.
    """

    def __init__(self):
        self.history: list[dict] = []
        self.collected: dict = {}
        self.prediction_done: bool = False
        self.last_prediction: Optional[dict] = None

    def _add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def start(self) -> str:
        self._add("assistant", GREETING)
        return GREETING

    # Keywords that signal the agent wants to assess a new customer
    NEW_CUSTOMER_SIGNALS = [
        "another customer", "new customer", "next customer", "different customer",
        "let's move on", "move on", "start over", "fresh", "new one",
        "another one", "someone else", "next one",
    ]

    def _is_new_customer_request(self, message: str) -> bool:
        msg = message.lower()
        return any(signal in msg for signal in self.NEW_CUSTOMER_SIGNALS)

    def respond(self, user_message: str) -> str:
        # ── Auto-reset if agent wants to discuss a new customer ──────────────
        if self.prediction_done and self._is_new_customer_request(user_message):
            self.__init__()
            greeting = (
                "Sure! Let's assess the next customer. "
                "Please share their details and I'll ask for anything I still need."
            )
            self._add("assistant", greeting)
            return greeting

        self._add("user", user_message)

        # 1. Extract features from the full conversation
        self.collected = extract_features_from_conversation(self.history)

        # 2. Check what is still missing
        missing = get_missing_fields(self.collected)

        if missing:
            # 3a. Build a prompt that explicitly shows what IS collected
            #     so the LLM cannot possibly re-ask for those fields
            collected_summary = ", ".join(
                f"{k}={v}" for k, v in self.collected.items()
            ) or "nothing yet"

            prompt = (
                f"Already collected (DO NOT ask for these again): {collected_summary}\n\n"
                f"Still missing (ask for these only):\n"
                + "\n".join(f"- {m}" for m in missing[:3])
                + "\n\nWrite a short, friendly message asking only for the missing fields above."
            )
            messages = [
                {"role": "system", "content": COLLECTION_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            reply = _call_groq(messages)

        else:
            # 3b. All features collected — run the ML model
            feature_vector = encode_collected_features(self.collected)

            if feature_vector is None:
                # Safety fallback: re-check which OHE group might be missing
                missing_ohe = [
                    g for g in ("internet_service", "payment_method")
                    if g not in self.collected
                ]
                reply = (
                    "Almost there! Could you clarify: "
                    + " and ".join(missing_ohe) + "?"
                )
            else:
                self.last_prediction = predict_churn(feature_vector)
                self.prediction_done = True
                reply = explain_result(
                    self.last_prediction, self.collected, self.history)

        self._add("assistant", reply)
        return reply

    def reset(self):
        """Start a fresh session."""
        self.__init__()
        return self.start()
