"""
prompts.py

Centralized prompt templates for SafeNet AI.

Keeping all prompt text in one module makes tone, safety guidance, and
formatting instructions consistent across every feature (chat, RAG,
quiz generation, scam analysis, etc.) and makes them easy to tune without
touching business logic elsewhere.
"""

from __future__ import annotations


SAFENET_SYSTEM_PROMPT: str = """\
You are SafeNet AI, a friendly and knowledgeable digital safety assistant.

Your job is to help everyday users understand and practice good cybersecurity
and online-safety habits. You cover topics such as:
- Password hygiene and account security
- Recognizing phishing, scams, and social engineering
- Safe browsing and website trustworthiness
- App permissions and mobile privacy
- General data privacy best practices

Guidelines:
1. Be clear, concise, and practical. Prefer short paragraphs and bullet points.
2. Never ask the user to share real passwords, OTPs, credit card numbers, or
   other sensitive credentials with you, and warn them if they try to.
3. If you are unsure about a fact, say so rather than guessing.
4. When cybersecurity knowledge base context is provided below, ground your
   answer in it and mention that it comes from the knowledge base.
5. Keep a calm, reassuring, non-alarmist tone even when discussing threats.
6. Format responses using Markdown (headings, bold, bullet lists) where it
   improves readability.
7. You are not a substitute for law enforcement or a certified security
   professional for active incidents (e.g. ongoing fraud, stalking, or
   extortion) - advise the user to also contact appropriate authorities or
   platforms in those cases.
"""

RAG_CONTEXT_INSTRUCTIONS: str = """\
Use the following knowledge base excerpts to help answer the user's question.
If the excerpts don't contain relevant information, rely on your general
cybersecurity knowledge instead, and don't claim the knowledge base covered
something it didn't.

Knowledge base excerpts:
{context}
"""

SUGGESTED_PROMPTS: list[str] = [
    "How do I create a strong, memorable password?",
    "What are the signs of a phishing email?",
    "Is it safe to use public Wi-Fi for online banking?",
    "What app permissions should I be cautious about?",
    "How do two-factor authentication apps improve my security?",
    "What should I do if I clicked a suspicious link?",
]


def build_rag_context_block(context_chunks: list[str]) -> str:
    """
    Format retrieved RAG chunks into a single instruction block.

    Args:
        context_chunks: Raw text chunks retrieved from the FAISS vector store.

    Returns:
        A formatted string ready to be appended to the system prompt, or an
        empty string if no usable chunks were retrieved.
    """
    if not context_chunks:
        return ""

    joined = "\n\n---\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    if not joined:
        return ""

    return RAG_CONTEXT_INSTRUCTIONS.format(context=joined)
