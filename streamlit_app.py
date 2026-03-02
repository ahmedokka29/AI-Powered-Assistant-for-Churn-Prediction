"""
streamlit_app.py
────────────────────────────────────────────────────────────────────────────────
Streamlit interface for the Churn Prediction Chatbot API.

Run:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit requests
    The FastAPI server must be running:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
────────────────────────────────────────────────────────────────────────────────
"""

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111827; }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 6px; }

    /* Result metric cards */
    [data-testid="stMetric"] { background: #1a2235; border-radius: 10px; padding: 12px; }

    /* Input box */
    [data-testid="stChatInput"] textarea { background: #1a2235 !important; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_FIELDS = [
    "gender", "senior_citizen", "is_married", "dependents", "tenure",
    "phone_service", "dual", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method", "monthly_charges", "total_charges",
]

TIER_CONFIG = {
    "LOW":    {"color": "#00ffaa", "emoji": "🟢", "label": "Low Risk"},
    "MEDIUM": {"color": "#ffd166", "emoji": "🟡", "label": "Medium Risk"},
    "HIGH":   {"color": "#ff6b6b", "emoji": "🔴", "label": "High Risk"},
}

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "session_id":      None,
    # list of {"role": "user"|"assistant", "content": str}
    "messages":        [],
    "missing_fields":  ALL_FIELDS.copy(),
    "prediction_done": False,
    "churn_prob":      None,
    "risk_tier":       None,
    "collected":       {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers ───────────────────────────────────────────────────────────────
def api_url(path: str) -> str:
    base = st.session_state.get(
        "api_base", "http://localhost:8000").rstrip("/")
    return f"{base}{path}"


def check_health() -> bool:
    try:
        r = requests.get(api_url("/health"), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_session() -> tuple[str, str] | None:
    """Start a new session. Returns (session_id, greeting) or None on error."""
    try:
        r = requests.post(api_url("/chat/start"), timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["session_id"], data["message"]
    except Exception as e:
        st.error(f"Could not start session: {e}")
        return None


def send_message(session_id: str, message: str) -> dict | None:
    try:
        r = requests.post(
            api_url("/chat/message"),
            json={"session_id": session_id, "message": message},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def reset_session(session_id: str) -> str | None:
    try:
        r = requests.post(api_url(f"/chat/reset/{session_id}"), timeout=10)
        r.raise_for_status()
        return r.json()["message"]
    except Exception as e:
        st.error(f"Reset failed: {e}")
        return None


def get_status(session_id: str) -> dict | None:
    try:
        r = requests.get(api_url(f"/chat/status/{session_id}"), timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")

    # API config
    st.markdown("### ⚙️ API Config")
    api_base = st.text_input(
        "Base URL", value="http://localhost:8000", key="api_base", label_visibility="collapsed"
    )

    # Connection status
    healthy = check_health()
    if healthy:
        st.success("✅ API connected")
    else:
        st.error("❌ API unreachable — is uvicorn running?")

    st.markdown("---")

    # Session controls
    st.markdown("### 🗂️ Session")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "▶ New Session", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True,
                              disabled=st.session_state.session_id is None)

    if st.session_state.session_id:
        sid = st.session_state.session_id
        st.markdown(
            f"<div style='font-size:10px;color:#5a6a85;word-break:break-all;"
            f"background:#0b0f1a;padding:8px;border-radius:6px;font-family:monospace'>"
            f"<b style='color:#00d4ff'>{sid[:8]}</b>{sid[8:]}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Field progress tracker
    st.markdown("### 📋 Fields Collected")
    missing_set = set(st.session_state.missing_fields)
    for field in ALL_FIELDS:
        done = field not in missing_set and st.session_state.session_id is not None
        icon = "✅" if done else "⬜"
        color = "#00ffaa" if done else "#5a6a85"
        st.markdown(
            f"<div style='font-size:11px;color:{color};font-family:monospace;"
            f"padding:2px 0'>{icon} {field.replace('_', ' ')}</div>",
            unsafe_allow_html=True,
        )

    # Prediction result card
    if st.session_state.prediction_done and st.session_state.churn_prob is not None:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        tier = st.session_state.risk_tier
        cfg = TIER_CONFIG.get(tier, TIER_CONFIG["MEDIUM"])
        prob_pct = int(st.session_state.churn_prob * 100)

        st.markdown(
            f"<div style='background:#1a2235;border-radius:12px;padding:16px;"
            f"border:1px solid {cfg['color']}44'>"
            f"<div style='font-size:40px;font-weight:800;color:{cfg['color']}'>{prob_pct}%</div>"
            f"<div style='font-size:13px;font-weight:700;color:{cfg['color']};letter-spacing:2px'>"
            f"{cfg['emoji']} {cfg['label'].upper()}</div>"
            f"<div style='margin-top:10px;background:{cfg['color']}22;border-radius:8px;height:8px'>"
            f"<div style='width:{prob_pct}%;height:100%;background:{cfg['color']};"
            f"border-radius:8px'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Handle button actions ─────────────────────────────────────────────────────
if start_btn:
    if not healthy:
        st.error("Cannot start session — API is not reachable.")
    else:
        result = start_session()
        if result:
            sid, greeting = result
            st.session_state.session_id = sid
            st.session_state.messages = [
                {"role": "assistant", "content": greeting}]
            st.session_state.missing_fields = ALL_FIELDS.copy()
            st.session_state.prediction_done = False
            st.session_state.churn_prob = None
            st.session_state.risk_tier = None
            st.session_state.collected = {}
            st.rerun()

if reset_btn and st.session_state.session_id:
    greeting = reset_session(st.session_state.session_id)
    if greeting:
        st.session_state.messages = [
            {"role": "assistant", "content": greeting}]
        st.session_state.missing_fields = ALL_FIELDS.copy()
        st.session_state.prediction_done = False
        st.session_state.churn_prob = None
        st.session_state.risk_tier = None
        st.session_state.collected = {}
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("## 💬 Churn Prediction Chat")
st.markdown(
    "<p style='color:#5a6a85;margin-top:-12px'>Powered by Random Forest · Llama 3.3 70B via Groq Cloud</p>",
    unsafe_allow_html=True,
)

if not st.session_state.session_id:
    # Empty state
    st.markdown(
        "<div style='text-align:center;padding:80px 0;color:#5a6a85'>"
        "<div style='font-size:48px'>🤖</div>"
        "<h3 style='color:#e8edf5'>Ready to assess</h3>"
        "<p>Click <b>▶ New Session</b> in the sidebar to begin a customer assessment.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Describe the customer or answer questions...",
        disabled=not st.session_state.session_id,
    )

    if user_input:
        # Show user message immediately
        st.session_state.messages.append(
            {"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Call API with spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                data = send_message(st.session_state.session_id, user_input)

            if data:
                reply = data["reply"]
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply})

                # Update sidebar field tracker from the status endpoint.
                # We always call /chat/status so field tracking is exact —
                # based on collected_features keys, not fragile label parsing.
                status = get_status(st.session_state.session_id)
                if status:
                    collected_keys = set(
                        status.get("collected_features", {}).keys())
                    # Map OHE group names back to ALL_FIELDS entries
                    # (status uses "internet_service", "payment_method" — same as ALL_FIELDS)
                    st.session_state.missing_fields = [
                        f for f in ALL_FIELDS if f not in collected_keys
                    ]
                    st.session_state.collected = status.get(
                        "collected_features", {})

                if data.get("prediction_done"):
                    st.session_state.prediction_done = True
                    st.session_state.churn_prob = data.get("churn_probability")
                    st.session_state.risk_tier = data.get("risk_tier")
                    st.session_state.missing_fields = []

                st.rerun()
