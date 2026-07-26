"""
app.py

Main Streamlit entry point for SafeNet AI - Digital Safety Assistant.

This phase implements:
    - Application shell: page config, dark theme injection, sidebar navigation
    - Home dashboard with feature cards
    - Fully working AI Chat page (Gemini + optional RAG grounding)
    - Placeholder ("coming soon") states for pages built in later phases

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chatbot import ChatBot, ChatBotError
from config import settings
from database import Database, DatabaseError
from prompts import SUGGESTED_PROMPTS
from rag import RAGPipeline, RAGPipelineError
from services.learning_service import LEARNING_MODULES
from services.quiz_service import QuizService
from services.scam_service import ScamAnalysisResult, ScamAnalysisService
from services.tip_service import TipService
from services.website_service import WebsiteAnalysisResult, WebsiteAnalysisService
from utils.helpers import (
    copy_button,
    load_css,
    render_empty_state,
    render_typing_indicator,
    stream_text,
)
from utils.logger import get_logger
from utils.website_checker import WebsiteCheckerError

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("home", "🏠", "Home"),
    ("chat", "💬", "Chat"),
    ("learn", "📚", "Learn"),
    ("scam", "🚨", "Scam Analyzer"),
    ("website", "🌐", "Website Safety"),
    ("quiz", "📝", "Quiz"),
    ("dashboard", "📊", "Dashboard"),
    ("settings", "⚙️", "Settings"),
]

# Pages implemented in this phase; the rest render a clean "coming soon" state.
IMPLEMENTED_PAGES = {"home", "chat", "scam", "website", "learn", "quiz", "dashboard"}


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
def init_session_state() -> None:
    """Set up default Streamlit session state values on first load."""
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if "chat_display" not in st.session_state:
        # UI-facing message list: mirrors ChatBot memory but also tracks
        # per-message ids/timestamps/feedback for rendering purposes.
        st.session_state.chat_display = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.chatbot_error = None

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "scam_last_result" not in st.session_state:
        st.session_state.scam_last_result = None

    if "website_last_result" not in st.session_state:
        st.session_state.website_last_result = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> Optional[RAGPipeline]:
    """
    Build or load the RAG pipeline once per server process and cache it.

    Returns:
        A ready RAGPipeline, or None if it could not be initialized (e.g.
        the embedding model can't be downloaded, or no PDFs exist yet).
        Chat still works without RAG - it just skips knowledge-base grounding.
    """
    try:
        pipeline = RAGPipeline()
        pipeline.ensure_index_ready()
        return pipeline
    except RAGPipelineError as exc:
        logger.warning(f"RAG pipeline unavailable, continuing without it: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_database() -> Database:
    """Build the SQLite Database connection layer once per server process."""
    return Database()


@st.cache_resource(show_spinner=False)
def get_quiz_service() -> QuizService:
    """Build the QuizService once per server process, backed by the shared Database."""
    return QuizService(get_database())


@st.cache_resource(show_spinner=False)
def get_tip_service() -> TipService:
    """Build the TipService once per server process, backed by the shared Database."""
    return TipService(get_database())


@st.cache_resource(show_spinner=False)
def get_scam_service() -> ScamAnalysisService:
    """
    Build the Scam Analysis service once per server process and cache it.

    The service itself holds no per-user state - it lazily initializes a
    Gemini client on first use and remembers whether that succeeded, so
    sharing one instance across sessions is safe.
    """
    return ScamAnalysisService()


@st.cache_resource(show_spinner=False)
def get_website_service() -> WebsiteAnalysisService:
    """
    Build the Website Analysis service once per server process and cache it.
    Safe to share across sessions - see get_scam_service() for the same
    lazy-init rationale.
    """
    return WebsiteAnalysisService()


def get_chatbot() -> Optional[ChatBot]:
    """
    Lazily create (once per session) and return the ChatBot instance,
    storing any initialization error for display instead of crashing the app.
    """
    if st.session_state.chatbot is not None:
        return st.session_state.chatbot

    try:
        st.session_state.chatbot = ChatBot()
        st.session_state.chatbot_error = None
    except ChatBotError as exc:
        st.session_state.chatbot = None
        st.session_state.chatbot_error = str(exc)

    return st.session_state.chatbot


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the fixed sidebar with branding and page navigation buttons."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0 0 1rem 0;">
                <span style="font-family:var(--font-display); font-size:1.3rem;
                             font-weight:700; color:var(--text-primary);">
                    🛡️ {settings.app_name}
                </span><br/>
                <span style="font-family:var(--font-mono); font-size:0.72rem;
                             color:var(--text-muted);">
                    v{settings.app_version} · Digital Safety Assistant
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for page_id, icon, label in NAV_ITEMS:
            is_active = st.session_state.page == page_id
            css_class = "nav-button-active" if is_active else "nav-button"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
                st.session_state.page = page_id
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Home page
# ----------------------------------------------------------------------
def render_home() -> None:
    """Render the Home dashboard: hero banner + feature card grid."""
    st.markdown(
        """
        <div class="safenet-hero">
            <div class="safenet-hero-title">Stay safe online, one smart decision at a time.</div>
            <p class="safenet-hero-subtitle">
                SafeNet AI combines a Gemini-powered assistant with a grounded
                cybersecurity knowledge base to help you spot scams, harden your
                passwords, and browse with confidence.
            </p>
            <div class="safenet-status-pill">
                <span class="radar-dot"></span> System Protected
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tip_service = get_tip_service()
    today_tip = tip_service.get_today_tip()
    st.markdown(
        f"""
        <div class="risk-flag-item" style="border-left-color:var(--brand-solid);">
            <div class="risk-flag-label">💡 Cyber Tip of the Day</div>
            <div class="risk-flag-desc">{today_tip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("💬", "AI Chat", "Ask anything about online safety and get grounded, plain-language answers.", "chat"),
        ("📚", "Learn", "Structured lessons covering the fundamentals of digital safety.", "learn"),
        ("🚨", "Scam Analyzer", "Paste a suspicious message to check for phishing/scam patterns.", "scam"),
        ("🌐", "Website Safety", "Get a heuristic safety read on a URL before you click through.", "website"),
        ("📝", "Quiz", "Test your cybersecurity knowledge with a quick interactive quiz.", "quiz"),
        ("📊", "Dashboard", "See your usage insights and safety-check history at a glance.", "dashboard"),
    ]

    for row_start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (icon, title, desc, target) in zip(cols, cards[row_start : row_start + 4]):
            with col:
                st.markdown(
                    f"""
                    <div class="safenet-card">
                        <div class="safenet-card-icon">{icon}</div>
                        <div class="safenet-card-title">{title}</div>
                        <div class="safenet-card-desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Open →", key=f"home_card_{target}", use_container_width=True):
                    st.session_state.page = target
                    st.rerun()


# ----------------------------------------------------------------------
# Chat page
# ----------------------------------------------------------------------
def render_chat_message(role: str, content: str, msg_id: str) -> None:
    """Render a single stored chat message as a themed bubble with actions."""
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
    label = "You" if role == "user" else "SafeNet AI"

    st.markdown(
        f"""
        <div class="chat-bubble {bubble_class}">
            <div class="chat-role-label">{label}</div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if role == "assistant":
        action_cols = st.columns([1, 1, 1, 9])
        with action_cols[0]:
            copy_button(content, key=f"copy_{msg_id}")
        with action_cols[1]:
            st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
            if st.button("👍", key=f"up_{msg_id}"):
                _save_feedback(content, "up")
                st.toast("Thanks for the feedback!")
            st.markdown("</div>", unsafe_allow_html=True)
        with action_cols[2]:
            st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
            if st.button("👎", key=f"down_{msg_id}"):
                _save_feedback(content, "down")
                st.toast("Thanks - we'll use this to improve.")
            st.markdown("</div>", unsafe_allow_html=True)


def _save_feedback(content: str, rating: str) -> None:
    """Persist a thumbs up/down on an assistant message, failing silently on DB errors."""
    try:
        get_database().add_feedback(st.session_state.session_id, content, rating)
    except DatabaseError as exc:
        logger.warning(f"Could not save feedback (continuing anyway): {exc}")


def _build_chat_transcript() -> str:
    """Format the current session's chat_display messages as a Markdown transcript."""
    lines = [f"# {settings.app_name} - Chat Transcript", ""]
    for msg in st.session_state.chat_display:
        speaker = "You" if msg["role"] == "user" else "SafeNet AI"
        timestamp = msg["timestamp"].strftime("%Y-%m-%d %I:%M %p")
        lines.append(f"**{speaker}** _{timestamp}_")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def render_chat() -> None:
    """Render the full AI Chat page: history, suggested prompts, and input."""
    header_cols = st.columns([3, 1, 1])
    with header_cols[0]:
        st.markdown("### 💬 AI Chat")
    with header_cols[1]:
        if st.session_state.chat_display:
            st.download_button(
                "⬇️ Export Chat",
                data=_build_chat_transcript(),
                file_name=f"safenet_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="export_chat_btn",
                use_container_width=True,
            )
    with header_cols[2]:
        if st.session_state.chat_display and st.button(
            "🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True
        ):
            st.session_state.chat_display = []
            chatbot = get_chatbot()
            if chatbot is not None:
                chatbot.reset()
            try:
                get_database().clear_chat_history(st.session_state.session_id)
            except DatabaseError as exc:
                logger.warning(f"Could not clear persisted chat history: {exc}")
            st.rerun()

    st.caption("Ask about passwords, phishing, privacy, or anything digital-safety related.")

    if not settings.gemini_api_key:
        st.warning(
            "No Gemini API key configured yet. Add `GEMINI_API_KEY` to your `.env` "
            "file (see `.env.example`) to enable live responses.",
            icon="⚠️",
        )

    chatbot = get_chatbot()
    if chatbot is None and st.session_state.chatbot_error:
        st.error(st.session_state.chatbot_error, icon="🚫")

    rag_pipeline = get_rag_pipeline()
    if rag_pipeline is not None and rag_pipeline.is_ready:
        st.caption("📚 Knowledge base is loaded - answers may be grounded in indexed PDFs.")

    # --- Suggested prompts (shown until the first message is sent) ---
    if not st.session_state.chat_display:
        st.markdown("**Try asking:**")
        prompt_cols = st.columns(3)
        for i, suggestion in enumerate(SUGGESTED_PROMPTS[:6]):
            with prompt_cols[i % 3]:
                st.markdown('<div class="suggested-prompt">', unsafe_allow_html=True)
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.pending_prompt = suggestion
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr class="safenet-divider"/>', unsafe_allow_html=True)

    # --- Render existing conversation ---
    for msg in st.session_state.chat_display:
        render_chat_message(msg["role"], msg["content"], msg["id"])

    # --- Input: either from a clicked suggestion or the chat box ---
    user_input = st.chat_input("Type your question about online safety...")
    if st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        _handle_user_message(user_input, chatbot, rag_pipeline)


def _handle_user_message(
    user_input: str, chatbot: Optional[ChatBot], rag_pipeline: Optional[RAGPipeline]
) -> None:
    """Process one submitted chat message: display, generate, stream, store."""
    user_msg_id = uuid.uuid4().hex[:8]
    st.session_state.chat_display.append(
        {"role": "user", "content": user_input, "id": user_msg_id, "timestamp": datetime.now()}
    )
    render_chat_message("user", user_input, user_msg_id)
    _save_chat_message("user", user_input)

    if chatbot is None:
        error_text = st.session_state.chatbot_error or "Chat is currently unavailable."
        st.error(error_text, icon="🚫")
        return

    typing_placeholder = st.empty()
    render_typing_indicator(typing_placeholder)

    context_chunks: list[str] = []
    if rag_pipeline is not None and rag_pipeline.is_ready:
        try:
            context_chunks = rag_pipeline.retrieve(user_input)
        except RAGPipelineError as exc:
            logger.warning(f"RAG retrieval failed, continuing without context: {exc}")

    try:
        reply = chatbot.send_message(user_input, context_chunks=context_chunks)
    except ChatBotError as exc:
        typing_placeholder.empty()
        st.error(str(exc), icon="🚫")
        return

    typing_placeholder.empty()
    reply_placeholder = st.empty()
    stream_text(reply_placeholder, reply)

    assistant_msg_id = uuid.uuid4().hex[:8]
    st.session_state.chat_display.append(
        {
            "role": "assistant",
            "content": reply,
            "id": assistant_msg_id,
            "timestamp": datetime.now(),
        }
    )
    _save_chat_message("assistant", reply)
    st.rerun()


def _save_chat_message(role: str, content: str) -> None:
    """Persist a chat message to the database, failing silently on DB errors."""
    try:
        get_database().add_chat_message(st.session_state.session_id, role, content)
    except DatabaseError as exc:
        logger.warning(f"Could not persist chat message (continuing anyway): {exc}")


# ----------------------------------------------------------------------
# Scam Message Analyzer page
# ----------------------------------------------------------------------
def render_scam() -> None:
    """Render the Scam Message Analyzer page: input, heuristics, AI explanation."""
    st.markdown("### 🚨 Scam Message Analyzer")
    st.caption(
        "Paste a suspicious SMS, email, or chat message to check it for common "
        "scam patterns. Nothing you paste here is stored or logged."
    )

    message = st.text_area(
        "Paste the message you received",
        height=160,
        placeholder=(
            "e.g. Dear customer, your account has been suspended. "
            "Verify your details immediately: http://bit.ly/xyz"
        ),
        key="scam_input",
    )

    use_ai = st.checkbox(
        "Include AI-generated explanation (uses Gemini)", value=True, key="scam_use_ai"
    )

    analyze_clicked = st.button("Analyze Message", type="primary", key="scam_analyze_btn", use_container_width=True)

    if analyze_clicked:
        if not message.strip():
            st.info("Paste a message above before analyzing.", icon="ℹ️")
        else:
            service = get_scam_service()
            with st.spinner("Analyzing message..."):
                result = service.analyze(message, use_ai_explanation=use_ai)
            st.session_state.scam_last_result = result

    if st.session_state.get("scam_last_result") is not None:
        st.markdown('<hr class="safenet-divider"/>', unsafe_allow_html=True)
        _render_scam_result(st.session_state.scam_last_result)


def _render_scam_result(result: ScamAnalysisResult) -> None:
    """Render the risk badge, score bar, detected flags, and AI explanation."""
    risk = result.risk_result

    st.markdown(
        f"""
        <div class="risk-badge" style="color:{risk.color}; background:{risk.background};">
            ⚠️ Risk Level: {risk.level.value} &nbsp;·&nbsp; Score {risk.score}/100
        </div>
        <div class="risk-score-bar-track">
            <div class="risk-score-bar-fill" style="width:{risk.score}%; background:{risk.color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not risk.flags:
        st.success(
            "No common scam patterns were detected in this message. "
            "Still, stay cautious with any unexpected request."
        )
    else:
        st.markdown("**Detected indicators:**")
        for flag in risk.flags:
            st.markdown(
                f"""
                <div class="risk-flag-item">
                    <div class="risk-flag-label">🚩 {flag.label}</div>
                    <div class="risk-flag-desc">{flag.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result.ai_available and result.ai_explanation:
        st.markdown("**🤖 AI Explanation:**")
        st.markdown(result.ai_explanation)
    elif not result.ai_available:
        st.caption("AI explanation unavailable right now - showing heuristic results only.")


# ----------------------------------------------------------------------
# Website Safety Advisor page
# ----------------------------------------------------------------------
def render_website() -> None:
    """Render the Website Safety Advisor page: URL input, heuristics, AI explanation."""
    st.markdown("### 🌐 Website Safety Advisor")
    st.caption(
        "Paste a URL to get a structural safety read before you click through. "
        "This checks the link itself only - SafeNet AI never visits or fetches the site."
    )

    url_input = st.text_input(
        "Enter a URL",
        placeholder="e.g. http://paypal-secure-login.tk/verify",
        key="website_input",
    )

    use_ai = st.checkbox(
        "Include AI-generated explanation (uses Gemini)", value=True, key="website_use_ai"
    )

    analyze_clicked = st.button("Analyze Website", type="primary", key="website_analyze_btn", use_container_width=True)

    if analyze_clicked:
        if not url_input.strip():
            st.info("Enter a URL above before analyzing.", icon="ℹ️")
        else:
            service = get_website_service()
            try:
                with st.spinner("Analyzing URL..."):
                    result = service.analyze(url_input, use_ai_explanation=use_ai)
                st.session_state.website_last_result = result
            except WebsiteCheckerError as exc:
                st.session_state.website_last_result = None
                st.error(str(exc), icon="🚫")

    if st.session_state.get("website_last_result") is not None:
        st.markdown('<hr class="safenet-divider"/>', unsafe_allow_html=True)
        _render_website_result(st.session_state.website_last_result)


def _render_website_result(result: WebsiteAnalysisResult) -> None:
    """Render the risk badge, score bar, detected flags, and AI explanation."""
    risk = result.risk_result

    st.markdown(f"**Analyzed:** `{result.url}`")
    st.markdown(
        f"""
        <div class="risk-badge" style="color:{risk.color}; background:{risk.background};">
            ⚠️ Risk Level: {risk.level.value} &nbsp;·&nbsp; Score {risk.score}/100
        </div>
        <div class="risk-score-bar-track">
            <div class="risk-score-bar-fill" style="width:{risk.score}%; background:{risk.color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not risk.flags:
        st.success(
            "No common structural red flags were detected for this URL. "
            "Still, always verify sensitive login pages by typing the address yourself."
        )
    else:
        st.markdown("**Detected indicators:**")
        for flag in risk.flags:
            st.markdown(
                f"""
                <div class="risk-flag-item">
                    <div class="risk-flag-label">🚩 {flag.label}</div>
                    <div class="risk-flag-desc">{flag.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result.ai_available and result.ai_explanation:
        st.markdown("**🤖 AI Explanation:**")
        st.markdown(result.ai_explanation)
    elif not result.ai_available:
        st.caption("AI explanation unavailable right now - showing heuristic results only.")


# ----------------------------------------------------------------------
# Learning Mode page
# ----------------------------------------------------------------------
def render_learn() -> None:
    """Render the Cybersecurity Learning Mode: structured, expandable lessons."""
    st.markdown("### 📚 Cybersecurity Learning Mode")
    st.caption("Short, structured lessons covering the fundamentals of staying safe online.")

    for module in LEARNING_MODULES:
        with st.expander(f"{module.icon}  {module.title} — {module.summary}"):
            st.markdown(module.content)


# ----------------------------------------------------------------------
# Interactive Quiz page
# ----------------------------------------------------------------------
def render_quiz() -> None:
    """Render the interactive cybersecurity quiz: question flow, scoring, history."""
    st.markdown("### 📝 Interactive Quiz")
    st.caption("Test your cybersecurity knowledge with a quick 5-question quiz.")

    quiz_service = get_quiz_service()

    if st.session_state.quiz_questions is None:
        if st.button("Start Quiz", type="primary", key="quiz_start_btn"):
            st.session_state.quiz_questions = quiz_service.get_random_questions(5)
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()
        _render_quiz_history(quiz_service)
        return

    questions = st.session_state.quiz_questions

    if not st.session_state.quiz_submitted:
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i + 1}. {q.question}**")
                st.session_state.quiz_answers[i] = st.radio(
                    label=f"quiz_q_{i}",
                    options=list(range(len(q.options))),
                    format_func=lambda idx, opts=q.options: opts[idx],
                    key=f"quiz_radio_{i}",
                    label_visibility="collapsed",
                    index=None,
                )
                st.markdown("")
            submitted = st.form_submit_button("Submit Quiz", type="primary")

        if submitted:
            unanswered = [i for i in range(len(questions)) if st.session_state.quiz_answers.get(i) is None]
            if unanswered:
                st.warning("Please answer every question before submitting.", icon="⚠️")
            else:
                st.session_state.quiz_submitted = True
                score = sum(
                    1 for i, q in enumerate(questions)
                    if st.session_state.quiz_answers[i] == q.correct_index
                )
                quiz_service.save_attempt(st.session_state.session_id, score, len(questions))
                st.session_state.quiz_score = score
                st.rerun()
        return

    # --- Results view ---
    score = st.session_state.quiz_score
    total = len(questions)
    percentage = round((score / total) * 100)

    st.markdown(
        f"""
        <div class="safenet-hero" style="padding:1.5rem 2rem;">
            <div class="safenet-hero-title" style="font-size:1.5rem;">
                You scored {score}/{total} ({percentage}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, q in enumerate(questions):
        user_answer = st.session_state.quiz_answers.get(i)
        is_correct = user_answer == q.correct_index
        icon = "✅" if is_correct else "❌"
        st.markdown(
            f"""
            <div class="risk-flag-item" style="border-left-color:{'var(--accent-safe)' if is_correct else 'var(--accent-danger)'};">
                <div class="risk-flag-label">{icon} {i + 1}. {q.question}</div>
                <div class="risk-flag-desc">
                    Your answer: {q.options[user_answer]}<br/>
                    Correct answer: {q.options[q.correct_index]}<br/>
                    {q.explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Try Another Quiz", key="quiz_retry_btn"):
        st.session_state.quiz_questions = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()

    _render_quiz_history(quiz_service)


def _render_quiz_history(quiz_service: QuizService) -> None:
    """Show this session's past quiz attempts, if any."""
    history = quiz_service.get_history(st.session_state.session_id)
    if not history:
        return

    st.markdown('<hr class="safenet-divider"/>', unsafe_allow_html=True)
    st.markdown("**Your past attempts this session:**")
    for entry in history[:5]:
        pct = round((entry.score / entry.total) * 100)
        st.caption(f"{entry.timestamp.strftime('%I:%M %p')} — {entry.score}/{entry.total} ({pct}%)")


# ----------------------------------------------------------------------
# Dashboard & Analytics page
# ----------------------------------------------------------------------
def render_dashboard() -> None:
    """Render usage analytics: chat volume, quiz performance, and feedback sentiment."""
    st.markdown("### 📊 Dashboard")
    st.caption("Aggregate usage insights drawn from chat activity, quizzes, and feedback.")

    db = get_database()

    total_messages = db.count_all_chat_messages()
    avg_quiz_pct = db.get_average_quiz_percentage()
    feedback_summary = db.get_feedback_summary()
    total_feedback = feedback_summary["up"] + feedback_summary["down"]

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Chat Messages", total_messages)
    with kpi_cols[1]:
        st.metric("Avg. Quiz Score", f"{avg_quiz_pct}%" if avg_quiz_pct is not None else "—")
    with kpi_cols[2]:
        st.metric("Feedback Collected", total_feedback)
    with kpi_cols[3]:
        satisfaction = (
            round((feedback_summary["up"] / total_feedback) * 100)
            if total_feedback else None
        )
        st.metric("Satisfaction Rate", f"{satisfaction}%" if satisfaction is not None else "—")

    st.markdown('<hr class="safenet-divider"/>', unsafe_allow_html=True)

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("**Messages per day (last 14 days)**")
        per_day = db.get_messages_per_day(days=14)
        if not per_day:
            render_empty_state(
                "No chat activity yet",
                "Once conversations happen in the AI Chat tab, daily volume will appear here.",
                icon="💬",
            )
        else:
            df = pd.DataFrame(per_day, columns=["Date", "Messages"])
            fig = px.bar(df, x="Date", y="Messages")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8E9EA",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_traces(marker_color="#669DF6")
            st.plotly_chart(fig, width='stretch')

    with chart_cols[1]:
        st.markdown("**Feedback sentiment**")
        if total_feedback == 0:
            render_empty_state(
                "No feedback yet",
                "Thumbs up/down on chat replies will show up here once collected.",
                icon="👍",
            )
        else:
            fig2 = go.Figure(
                data=[
                    go.Pie(
                        labels=["👍 Helpful", "👎 Not helpful"],
                        values=[feedback_summary["up"], feedback_summary["down"]],
                        hole=0.55,
                        marker=dict(colors=["#4ADE80", "#EF4444"]),
                    )
                ]
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8E9EA",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
            )
            st.plotly_chart(fig2, width='stretch')

    st.markdown("**Quiz score history (this session)**")
    quiz_history = db.get_quiz_scores(st.session_state.session_id)
    if not quiz_history:
        render_empty_state(
            "No quizzes taken yet",
            "Complete a quiz in the Quiz tab to see your score trend here.",
            icon="📝",
        )
    else:
        quiz_df = pd.DataFrame(
            [
                {
                    "Attempt": i + 1,
                    "Score %": round((entry.score / entry.total) * 100),
                }
                for i, entry in enumerate(reversed(quiz_history))
            ]
        )
        fig3 = px.line(quiz_df, x="Attempt", y="Score %", markers=True)
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8E9EA",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_range=[0, 100],
        )
        fig3.update_traces(line_color="#A78BFA", marker_color="#A78BFA")
        st.plotly_chart(fig3, width='stretch')


# ----------------------------------------------------------------------
# Placeholder pages for later build phases
# ----------------------------------------------------------------------
_UPCOMING_PAGES: dict[str, tuple[str, str, str]] = {
    "password": ("🔐", "Password Strength Analyzer", "This tool is built in Phase 5."),
    "privacy": ("📱", "Privacy Permission Advisor", "This tool is built in Phase 8."),
    "settings": ("⚙️", "Settings", "Preferences and data controls are built in a later phase."),
}


def render_upcoming_page(page_id: str) -> None:
    """Render a clean, on-brand 'coming soon' state for a not-yet-built page."""
    icon, title, description = _UPCOMING_PAGES[page_id]
    render_empty_state(title=title, description=description, icon=icon)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    """Application entry point: init state, render theme, sidebar, and page."""
    init_session_state()
    load_css()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "chat":
        render_chat()
    elif page == "scam":
        render_scam()
    elif page == "website":
        render_website()
    elif page == "learn":
        render_learn()
    elif page == "quiz":
        render_quiz()
    elif page == "dashboard":
        render_dashboard()
    elif page in IMPLEMENTED_PAGES:
        pass  # reserved for future implemented pages
    else:
        render_upcoming_page(page)


if __name__ == "__main__":
    main()
