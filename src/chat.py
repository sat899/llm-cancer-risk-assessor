"""Conversational chat agent over the NG12 guidelines.

This module provides a RAG-based chat agent that:
1. Retrieves relevant guideline chunks from the vector store
2. Answers in natural language using only retrieved evidence
3. Cites specific guideline passages (page / section)
4. Maintains multi-turn conversation history per session (in-memory)

The chat agent does NOT use Gemini function calling — it performs RAG
retrieval server-side, injects passages into the prompt, and lets
Gemini reason over them.  This is simpler and more controllable for
a Q&A use case compared to the tool-calling pattern in ``agent.py``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import config
from src.tools import search_clinical_guidelines

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (see PROMPTS.md for design rationale)
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You are a clinical guideline assistant specialising in the NICE NG12 \
guidelines ("Suspected cancer: recognition and referral").

## Your Role
Answer questions about the NG12 cancer referral guidelines using ONLY the \
guideline passages provided in the CONTEXT section below.  You support \
multi-turn conversations — the user may ask follow-up questions that refer \
to earlier parts of the conversation.

## Rules
1. **Evidence-only** — base every answer on the CONTEXT passages.  Do NOT \
use general medical knowledge or information not present in the provided \
passages.
2. **Cite your sources** — whenever you make a clinical statement, include \
an inline citation in the format [NG12 p.XX].
3. **Refuse gracefully** — if the CONTEXT does not contain enough \
information to answer the question, say so clearly: "I couldn't find \
support in the NG12 text for that.  The retrieved passages cover: ..." \
and summarise what IS available.
4. **Do NOT invent** — never fabricate thresholds, age cut-offs, or \
referral criteria that are not explicitly stated in the CONTEXT.
5. **Be concise** — answer directly, then provide supporting detail.

## Output Format
Return a JSON object (no markdown fences, no surrounding text):

{
    "answer": "Your natural-language answer with [NG12 p.XX] citations.",
    "citations": [
        {
            "source": "NG12 PDF",
            "page": 0,
            "chunk_id": "chunk_0",
            "excerpt": "Short relevant excerpt from the passage"
        }
    ]
}
"""

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# Dict[session_id, List[dict]]  where each dict is {"role": ..., "content": ...}
_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return conversation history for a session."""
    return list(_sessions.get(session_id, []))


def clear_history(session_id: str) -> bool:
    """Clear conversation history.  Returns True if session existed."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# ---------------------------------------------------------------------------
# Chat agent
# ---------------------------------------------------------------------------

class ChatAgent:
    """Conversational RAG agent over the NG12 guidelines."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.VERTEX_AI_MODEL_NAME
        self._model = None

    def _initialize_model(self):
        """Lazy-init the Gemini model (no tools — just generation)."""
        if self._model is not None:
            return self._model

        from vertexai import init as vertex_init
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        vertex_init(
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION,
        )

        self._model = GenerativeModel(
            self.model_name,
            system_instruction=CHAT_SYSTEM_PROMPT,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        logger.info("Initialised chat model: %s", self.model_name)
        return self._model

    # -- public API ----------------------------------------------------------

    def chat(
        self, session_id: str, message: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Process a user message and return an answer with citations.

        Steps:
        1. Search vector store with the user message
        2. Build a prompt with: retrieved passages + conversation history + new message
        3. Send to Gemini
        4. Parse response, store in session, return

        Args:
            session_id: Client-generated session identifier.
            message: The user's question.
            top_k: Number of guideline chunks to retrieve.

        Returns:
            Dict with keys: session_id, answer, citations.
        """
        model = self._initialize_model()

        # 1. RAG retrieval
        passages = search_clinical_guidelines(message, top_k=top_k)
        context_block = self._format_context(passages)

        # 2. Build prompt parts
        history = _sessions[session_id]
        prompt = self._build_prompt(context_block, history, message)

        # 3. Generate
        response = model.generate_content(prompt)
        raw_text = response.text

        # 4. Parse
        parsed = self._parse_response(raw_text, passages)
        answer = parsed["answer"]
        citations = parsed["citations"]

        # 5. Store in session history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        return {
            "session_id": session_id,
            "answer": answer,
            "citations": citations,
        }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _format_context(passages: List[Dict[str, Any]]) -> str:
        """Format retrieved guideline passages into a CONTEXT block."""
        if not passages:
            return "CONTEXT:\nNo relevant guideline passages were found."

        lines = ["CONTEXT:"]
        for i, p in enumerate(passages):
            chunk_id = f"chunk_{i}"
            page = p.get("page_number", 0)
            section = p.get("section", "Unknown")
            content = p.get("content", "")
            lines.append(
                f"\n--- [{chunk_id}] Page {page} | {section} ---\n"
                f"{content[:3000]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_prompt(
        context: str, history: List[Dict[str, str]], new_message: str
    ) -> str:
        """Assemble the full prompt from context, history, and new message."""
        parts = [context, ""]

        if history:
            parts.append("CONVERSATION HISTORY:")
            for msg in history[-20:]:  # keep last 20 messages to limit context
                role = msg["role"].upper()
                parts.append(f"{role}: {msg['content']}")
            parts.append("")

        parts.append(f"USER QUESTION: {new_message}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(
        raw_text: str, passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Gemini's JSON response, with fallback for free-text."""
        try:
            json_str = raw_text
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]

            parsed = json.loads(json_str.strip())
            answer = parsed.get("answer", raw_text)
            citations = parsed.get("citations", [])

            # Normalise citation fields
            normalised = []
            for c in citations:
                normalised.append({
                    "source": c.get("source", "NG12 PDF"),
                    "page": int(c.get("page", 0)),
                    "chunk_id": c.get("chunk_id", "unknown"),
                    "excerpt": c.get("excerpt", "")[:500],
                })
            return {"answer": answer, "citations": normalised}

        except (json.JSONDecodeError, AttributeError):
            # Model returned free text — wrap it and build citations from passages
            logger.warning("Chat response was not valid JSON — using raw text")
            fallback_citations = [
                {
                    "source": "NG12 PDF",
                    "page": p.get("page_number", 0),
                    "chunk_id": f"chunk_{i}",
                    "excerpt": p.get("content", "")[:200],
                }
                for i, p in enumerate(passages[:3])
            ]
            return {"answer": raw_text, "citations": fallback_citations}
