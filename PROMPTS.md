## Prompt Engineering (brief overview)

This file documents the key ideas behind the two system prompts used in the system.

---

## High‑level comparison

- **Assessment agent**: multi‑turn, function‑calling, produces a structured risk‑assessment JSON object.
- **Chat agent**: single‑call, server‑retrieval, produces an answer‑with‑citations JSON object and can refuse when evidence is missing.

---

## Assessment agent (`SYSTEM_PROMPT` in `src/agent.py`)

- **Purpose**: Produce a cancer risk assessment grounded in **NICE NG12** with one of three outcomes: `Urgent Referral`, `Urgent Investigation`, or `Routine`.
- **Workflow**:
  1. **RETRIEVE**: Call `get_patient_data(patient_id)` to load demographics, symptoms, and history.
  2. **SEARCH**: Call `search_clinical_guidelines(query)` one or more times with targeted NG12 queries.
  3. **ANALYSE**: Match patient presentation against NG12 criteria (age, gender, smoking, duration, symptom combinations).
  4. **ASSESS**: Choose a category, breaking ties toward **higher urgency** (safety‑first).
- **Rules**:
  - Use **only** NG12 passages returned by `search_clinical_guidelines` (no general medical knowledge).
  - Always provide citations with page numbers and section titles.
  - Follow the JSON schema in the prompt so the backend can parse results reliably.
- **Model settings**: Temperature `0.1` for deterministic, clinically safe behaviour.
- **Rationale**:
  - Encourages **tool use first, judgment second**, so the model does not guess without patient data or guideline text.
  - The three fixed categories and JSON schema make downstream handling simple and safe.
  - Low temperature plus “when in doubt, choose higher urgency” encodes a conservative, clinical safety bias.

---

## Chat agent (`CHAT_SYSTEM_PROMPT` in `src/chat.py`)

- **Purpose**: Answer user questions about NG12 using **server‑side RAG** context, with citations.
- **Rules** (what the prompt enforces):
  - **Evidence‑only**: base answers solely on the provided CONTEXT passages.
  - **Citations**: include inline references in the form `[NG12 p.XX]`.
  - **No invention**: never fabricate thresholds, criteria, or page numbers.
  - **Refusal**: if the context does not support an answer, use a short refusal template instead of guessing.
  - **Concise**: give a direct answer first, then brief supporting detail.
- **Output**: JSON with an `answer` string (including inline citations) and a `citations` array (source, page, excerpt, etc.).
- **Model settings**: Temperature `0.2` for clearer, more natural prose while staying grounded.
- **Rationale**:
  - Keeps the model tightly **bound to retrieved context**, reducing hallucination risk.
  - Inline `[NG12 p.XX]` citations make answers auditable for clinicians.
  - Explicit refusal + concise style supports safe, fast Q&A rather than decision‑making.