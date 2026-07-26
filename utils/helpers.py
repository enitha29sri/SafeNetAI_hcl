"""
utils/helpers.py

Small, reusable UI helper functions shared across SafeNet AI's Streamlit
pages: theme loading, typing/streaming animation, copy-to-clipboard
buttons, and timestamp formatting. Centralizing these avoids duplicating
the same HTML/CSS snippets in every page module.
"""

from __future__ import annotations

import html
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import settings


def load_css(css_path: Path | None = None) -> None:
    """
    Inject the application's stylesheet into the current Streamlit page.

    Args:
        css_path: Path to the CSS file. Defaults to assets/css/style.css
            under the project root.
    """
    path = css_path or (settings.base_dir / "assets" / "css" / "style.css")
    if not path.exists():
        return

    css_text = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def stream_text(placeholder: "st.delta_generator.DeltaGenerator", text: str, delay: float = 0.012) -> None:
    """
    Reveal `text` progressively inside `placeholder` to simulate a typing
    animation, then render the final Markdown once fully revealed.

    Args:
        placeholder: An `st.empty()` slot to render into.
        text: The full response text to stream.
        delay: Seconds to wait between chunks. Kept small so long
            responses don't feel sluggish.
    """
    if not text:
        placeholder.markdown("")
        return

    # Reveal in small word-chunks rather than character-by-character:
    # noticeably smoother for longer cybersecurity explanations.
    words = text.split(" ")
    revealed = ""
    chunk_size = 3

    for i in range(0, len(words), chunk_size):
        revealed += (" " if revealed else "") + " ".join(words[i : i + chunk_size])
        placeholder.markdown(revealed + " ▌")
        time.sleep(delay)

    placeholder.markdown(text)


def render_typing_indicator(placeholder: "st.delta_generator.DeltaGenerator") -> None:
    """Show an animated three-dot 'typing' indicator inside `placeholder`."""
    placeholder.markdown(
        """
        <div class="chat-bubble chat-bubble-assistant">
            <div class="chat-role-label">SafeNet AI</div>
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def copy_button(text: str, key: str | None = None) -> None:
    """
    Render a small 'Copy' button that copies `text` to the clipboard via
    the browser's Clipboard API. Streamlit has no native copy widget, so
    this uses a lightweight inline HTML button instead of a heavier
    custom component.

    Args:
        text: The text to copy when clicked.
        key: Optional unique key; a random one is generated if omitted.
    """
    button_id = key or f"copy-{uuid.uuid4().hex[:8]}"
    # Escape for safe embedding inside an HTML attribute.
    escaped = html.escape(text, quote=True).replace("\n", "&#10;")

    st.markdown(
        f"""
        <div class="msg-action-btn">
            <button onclick="navigator.clipboard.writeText(this.getAttribute('data-text'));
                              this.innerText='✅ Copied';
                              setTimeout(() => this.innerText='📋 Copy', 1500);"
                    data-text="{escaped}"
                    id="{button_id}"
                    style="cursor:pointer; background:transparent; border:none;
                           color:var(--text-muted); font-size:0.78rem;">
                📋 Copy
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_timestamp(dt: datetime) -> str:
    """
    Format a datetime for display in chat history / exports.

    Args:
        dt: The timestamp to format.

    Returns:
        A human-readable string, e.g. "02:41 PM".
    """
    return dt.strftime("%I:%M %p")


def render_empty_state(title: str, description: str, icon: str = "🛠") -> None:
    """
    Render a consistent 'coming soon' / empty-state block for pages whose
    feature is implemented in a later build phase.

    Args:
        title: Short heading for the empty state.
        description: One or two sentences of explanation.
        icon: A single emoji shown above the title.
    """
    st.markdown(
        f"""
        <div class="safenet-empty-state">
            <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
