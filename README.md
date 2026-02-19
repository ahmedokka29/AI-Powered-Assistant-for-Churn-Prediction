# AI Powered Assistant for Churn Prediction — Documentation

## Overview

This system lets your marketing team assess customer churn risk through a **natural conversation** — no forms, no spreadsheets. An AI assistant gathers the customer's details through chat, runs a machine learning model trained on your historical data, and returns a plain-English risk assessment with an actionable recommendation.

```
Marketing Agent
     │
     │  natural language messages
     ▼
 FastAPI (api.py)
     │
     ├──► Groq Cloud — Llama 3.3 70B     ← extracts features, explains results
     │    (open-source, cloud-hosted)
     │
     └──► Random Forest Classifier        ← predicts churn probability
              Test AUC = 0.8356
              CV  AUC  = 0.8512
```

---

## Model Selection

Ten models were evaluated. Final leaderboard (ROC-AUC):

| Model                          | AUC        |
| ------------------------------ | ---------- |
| **Random Forest (tuned)** ✅    | **0.8512** |
| XGBoost (tuned)                | 0.8503     |
| Logistic Regression (baseline) | 0.8434     |
| CatBoost tuned (encoded)       | 0.8384     |
| CatBoost baseline (encoded)    | 0.8362     |
| CatBoost baseline (raw)        | 0.8359     |
| CatBoost tuned (raw)           | 0.8357     |
| Random Forest (baseline)       | 0.8340     |
| XGBoost (baseline)             | 0.8225     |
| Decision Tree (baseline)       | 0.8118     |

**Why Random Forest?** Best CV AUC (0.8512) and test AUC (0.8356). Top predictors align with domain knowledge: `contract type`, `tenure`, and `total_charges`. The model also provides interpretable feature importances, which helps explain results to the marketing team. Class imbalance (73% / 27%) was handled with `class_weight="balanced"` — no resampling.

**Best hyperparameters:** `n_estimators=300`, `max_depth=None`, `min_samples_leaf=10`

---

## Project Structure

```
project/
├── assets/
│   ├── data.csv
│   ├── encode_features.py        ← portable encoding (no sklearn objects)
│   └── model/
│       ├── churn_model.pkl       ← trained Random Forest
│       └── feature_names.json   ← column order for the model input vector
│
├── churn_classification.py       ← full modelling notebook / script
├── chatbot_pipeline.py           ← LLM + ML orchestration logic
├── api.py                        ← FastAPI REST endpoints
├── chat_tester.html              ← browser-based chat UI for testing
├── .env                          ← GROQ_API_KEY goes here (never commit this)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get your free Groq API key

Groq hosts open-source models (Llama 3.3, Mixtral, Gemma) in the cloud — no model downloads, no GPU, no local server needed. Inference runs at 300–800 tokens/second.

1. Sign up at **https://console.groq.com** (no credit card required)
2. Go to **API Keys → Create API Key**
3. Copy the key (starts with `gsk_...`)
4. Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

> **To swap models**, edit `GROQ_MODEL` in `chatbot_pipeline.py`:
>
> | Model ID | Speed | Best for |
> |---|---|---|
> | `llama-3.3-70b-versatile` | ~300 tok/s | Best quality (default) |
> | `llama3-8b-8192` | ~800 tok/s | Fastest responses |
> | `mixtral-8x7b-32768` | ~500 tok/s | Long multi-turn chats |

### 3. Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API is live. Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 4. Open the chat tester

Open `chat_tester.html` directly in your browser — no extra server needed. Click **Start New Session** to begin.

---

## Chat Tester UI

`chat_tester.html` is a self-contained browser interface for testing the API conversationally.

**Left sidebar:**
- API base URL field — change port if needed
- Live connection indicator (polls `/health` every 10 seconds — green when connected)
- **Start New Session** / **Reset Session** buttons
- Session ID display
- Field tracker — all 19 features listed, turn green ✅ as the bot collects them
- Prediction result card — appears when done, shows probability %, risk tier badge, and a colored progress bar

**Chat area:**
- Typing indicator (bouncing dots) while the LLM responds
- Messages animate in with fade-up
- Enter to send, Shift+Enter for new line
- After a prediction, click **Reset Session** in the sidebar to start a fresh customer assessment

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Liveness check.
```json
{ "status": "ok", "model": "RandomForest (AUC 0.85)", "llm": "groq/llama-3.3-70b-versatile" }
```

### `POST /chat/start`
Open a new conversation session. Returns a `session_id` — include it in all subsequent calls.
```json
{ "session_id": "3f8a2c1d-...", "message": "Hello! I'm your churn prediction assistant..." }
```

### `POST /chat/message`
Send a message. The bot extracts features, asks for missing ones, and runs the model once all 19 fields are collected.

**Request**
```json
{ "session_id": "3f8a2c1d-...", "message": "Male, 58 years old, senior citizen, married..." }
```

**Response — still collecting**
```json
{
  "session_id": "3f8a2c1d-...",
  "reply": "Does the customer have a device protection plan?",
  "prediction_done": false,
  "churn_probability": null,
  "risk_tier": null,
  "missing_fields": ["Device protection plan for internet devices? (yes / no / no_internet_service)"]
}
```

**Response — prediction complete**
```json
{
  "session_id": "3f8a2c1d-...",
  "reply": "This customer has an 85.8% chance of churning — HIGH risk...",
  "prediction_done": true,
  "churn_probability": 0.858,
  "risk_tier": "HIGH",
  "missing_fields": []
}
```

### `GET /chat/status/{session_id}`
Inspect collected features and session state mid-conversation. Useful for debugging or building a progress dashboard.

### `POST /chat/reset/{session_id}`
Wipe the session and start a fresh customer assessment. Use this (or the Reset button in the chat tester) when moving to a new customer.

---

## Risk Tiers

| Tier   | Churn Probability | Recommended Action                                             |
| ------ | ----------------- | -------------------------------------------------------------- |
| LOW    | < 40%             | No urgent action — standard engagement                         |
| MEDIUM | 40%–70%           | Proactive outreach — offer loyalty discount or service upgrade |
| HIGH   | ≥ 70%             | Priority retention — escalate to account manager immediately   |

---

## Customer Fields Reference

The model requires 19 fields. The bot collects them through conversation — the agent doesn't need to memorize this list.

| Field             | Accepted Values                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------- |
| Gender            | `male` / `female`                                                                         |
| Senior citizen    | `yes` / `no` (age ≥ 65 is inferred automatically)                                         |
| Married           | `yes` / `no`                                                                              |
| Dependents        | `yes` / `no`                                                                              |
| Tenure            | Number of months (0–72)                                                                   |
| Phone service     | `yes` / `no`                                                                              |
| Dual line         | `yes` / `no` / `no_phone_service`                                                         |
| Internet service  | `dsl` / `fiber_optic` / `no`                                                              |
| Online security   | `yes` / `no` / `no_internet_service`                                                      |
| Online backup     | `yes` / `no` / `no_internet_service`                                                      |
| Device protection | `yes` / `no` / `no_internet_service`                                                      |
| Tech support      | `yes` / `no` / `no_internet_service`                                                      |
| Streaming TV      | `yes` / `no` / `no_internet_service`                                                      |
| Streaming movies  | `yes` / `no` / `no_internet_service`                                                      |
| Contract          | `month_to_month` / `one_year` / `two_year`                                                |
| Paperless billing | `yes` / `no`                                                                              |
| Payment method    | `electronic_check` / `mailed_check` / `bank_transfer_automatic` / `credit_card_automatic` |
| Monthly charges   | Amount in $                                                                               |
| Total charges     | Total billed to date in $                                                                 |

The LLM infers several fields automatically from natural language — agents can speak freely:

| Natural phrasing                            | Inferred as                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| "fiber optic" or "fiber"                    | `internet_service = fiber_optic`                                |
| "pays by electronic check"                  | `payment_method = electronic_check` + `paperless_billing = yes` |
| "no phone service"                          | `phone_service = no` + `dual = no_phone_service`                |
| "streams TV and movies"                     | `streaming_tv = yes` + `streaming_movies = yes`                 |
| "senior citizen" or age ≥ 65                | `senior_citizen = 1`                                            |
| "month-to-month"                            | `contract = month_to_month`                                     |
| "no backup / no security / no tech support" | sets each respective field to `no`                              |

---

## Example Conversation

```
👤 Agent:  [Clicks "Start New Session"]

🤖 Bot:    Hello! I'm your churn prediction assistant. Please tell me about
           the customer — you can share details naturally and I'll ask
           follow-up questions for anything I still need.

👤 Agent:  Male customer, 58 years old, senior citizen. Been with us 12 months,
           married, has dependents, phone service, fiber optic internet.
           No online security or backup. No tech support. Streams both TV
           and movies. Month-to-month, pays by electronic check.
           $95/month, total so far $1,140.

🤖 Bot:    Does the customer have a device protection plan for his fiber
           optic devices? Also, does he have a second phone line?

👤 Agent:  No device protection. Yes, dual line.

🤖 Bot:    This customer has an 85.8% chance of churning — HIGH risk. The
           main drivers are his month-to-month contract and fiber optic plan
           with no add-ons. Recommend contacting him within 48 hours to
           offer a one-year contract with a bundled protection package.

           [Sidebar: 86% · HIGH RISK · red bar · all 19 fields green]

👤 Agent:  [Clicks "Reset Session" to assess the next customer]
```

---

## Encoding Strategy

All encoding lives in `encode_features.py` as plain Python dictionaries — no sklearn objects, nothing to serialize or version. Encoding is identical at training time and inference time.

| Type                | Features                                                                                                        | Method                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Binary              | gender, is_married, dependents, phone_service, paperless_billing                                                | 0 / 1 map                                                      |
| Ordinal (justified) | dual, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract | 0 / 1 / 2 — order is semantically real (no_service < no < yes) |
| One-Hot             | internet_service, payment_method                                                                                | OHE — genuinely nominal, no meaningful order                   |
| Numeric             | tenure, monthly_charges, total_charges, senior_citizen                                                          | Passed through as-is                                           |

---

## Troubleshooting

| Problem                     | Solution                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `GROQ_API_KEY is not set`   | Ensure `.env` is in the project root with `GROQ_API_KEY=gsk_...`                    |
| `401 Unauthorized`          | API key expired — generate a new one at console.groq.com                            |
| `429 Too Many Requests`     | Free tier rate limit hit — wait 60 seconds or upgrade                               |
| Chat tester can't reach API | Check base URL in the sidebar matches your uvicorn port; confirm uvicorn is running |
| `Session not found`         | Sessions are in-memory and clear on server restart — click Start New Session        |
| Low prediction accuracy     | Model was trained on a fixed dataset — retrain periodically on fresh data           |