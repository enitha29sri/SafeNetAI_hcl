"""
services/learning_service.py

Static content for SafeNet AI's Cybersecurity Learning Mode: short,
structured lessons covering the fundamentals of digital safety. Kept as
plain data (no external calls, no database) since this content is
curated and doesn't change at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LearningModule:
    """A single structured lesson shown in Learning Mode."""

    icon: str
    title: str
    summary: str
    content: str  # Markdown body


LEARNING_MODULES: list[LearningModule] = [
    LearningModule(
        icon="🔐",
        title="Password Fundamentals",
        summary="Why length beats complexity, and how password managers help.",
        content=(
            "A strong password is **long, unique, and unpredictable**. Length matters more "
            "than complexity — a 16-character passphrase like `correct-horse-battery-staple` "
            "can be both memorable and very hard to crack.\n\n"
            "**Key habits:**\n"
            "- Never reuse a password across sites — one breach shouldn't compromise everything.\n"
            "- Use a password manager to generate and store unique passwords per site.\n"
            "- Enable two-factor authentication (2FA) as a second layer of defense.\n"
            "- Change a password immediately if a service you use reports a breach."
        ),
    ),
    LearningModule(
        icon="🎣",
        title="Recognizing Phishing",
        summary="The tell-tale signs of scam emails, texts, and calls.",
        content=(
            "Phishing tricks you into revealing sensitive information by impersonating a "
            "trusted source. Common signs include:\n\n"
            "- **Urgency or threats** — \"act now or your account will be closed.\"\n"
            "- **Generic greetings** — \"Dear Customer\" instead of your real name.\n"
            "- **Mismatched links** — the visible text doesn't match where it actually leads.\n"
            "- **Requests for credentials** — legitimate organizations don't ask for your "
            "password or OTP via message.\n\n"
            "When in doubt, contact the organization directly using a phone number or website "
            "you already trust — not one provided in the suspicious message."
        ),
    ),
    LearningModule(
        icon="🌐",
        title="Safe Browsing Basics",
        summary="What to check before you click or enter information.",
        content=(
            "Before entering any sensitive information on a website:\n\n"
            "- Confirm the connection uses **HTTPS** (a padlock icon in the address bar).\n"
            "- Double-check the **domain name** carefully — phishing sites often use lookalike "
            "spellings.\n"
            "- Be cautious of shortened links (bit.ly, tinyurl) since the real destination is "
            "hidden until you click.\n"
            "- Avoid entering banking or login details while connected to public Wi-Fi without "
            "a VPN."
        ),
    ),
    LearningModule(
        icon="📱",
        title="App Permissions & Mobile Privacy",
        summary="What permissions actually mean, and when to say no.",
        content=(
            "Mobile apps often request more access than they need. A useful rule: **a "
            "permission should match the app's core function.**\n\n"
            "- A flashlight app doesn't need your contacts or microphone.\n"
            "- A messaging app reasonably needs contacts and microphone access.\n"
            "- Periodically review permissions in your phone's settings and revoke anything "
            "that no longer makes sense.\n"
            "- Be cautious of apps requesting broad accessibility access on Android — it's "
            "powerful and rarely necessary outside genuine accessibility tools."
        ),
    ),
    LearningModule(
        icon="🛡️",
        title="Account Security Layers",
        summary="Building defense in depth for your most important accounts.",
        content=(
            "No single measure is perfect, so security works best in layers:\n\n"
            "1. **Unique, strong password** for every account.\n"
            "2. **Two-factor authentication**, preferably via an authenticator app rather than "
            "SMS where possible.\n"
            "3. **Recovery options kept current** — an outdated recovery email or phone number "
            "can lock you out when you need it most.\n"
            "4. **Regular check-ins** on account activity logs for logins you don't recognize."
        ),
    ),
]
