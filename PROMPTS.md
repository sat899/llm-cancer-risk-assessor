# Prompt Engineering Strategy

This document explains the system prompt design for both agents in the Clinical Decision Support system.

---

## 1. Assessment Agent (`src/agent.py`)

### Location

`SYSTEM_PROMPT` constant in `src/agent.py`, passed via `system_instruction` at model init.

### Design Goals

| Goal | How the prompt achieves it |
|---|---|
| **Grounded in evidence** | The agent must base assessments on NG12 text returned by `search_clinical_guidelines`, not general medical knowledge. |
| **Structured output** | A strict JSON schema is defined in the prompt so the response can be parsed programmatically. |
| **Transparent reasoning** | The `reasoning` field forces the model to explain its clinical logic. |
| **Citation accuracy** | The prompt requires page numbers and section titles from the guideline passages. |
| **Safety-first** | "When in doubt, choose higher urgency" encodes the precautionary principle. |

### Prompt Structure: Role → Workflow → Rules → Output

#### 1. Role Definition
A concise identity statement naming the specific guideline (NG12) and task (risk assessment) to keep the model in domain.

#### 2. Workflow (Step-by-step)
```
1. RETRIEVE — Call get_patient_data(patient_id)
2. SEARCH  — Call search_clinical_guidelines(query)  (one or more times)
3. ANALYSE — Compare patient presentation against criteria
4. ASSESS  — Determine the risk category
```
Explicit ordering prevents the model from answering before gathering data. "One or more times" allows the model to cover multiple cancer pathways.

#### 3. Assessment Categories
Three tiers matching the NG12 document: Urgent Referral (2-week wait), Urgent Investigation, Routine.

#### 4. Rules
- Evidence-only reasoning
- Multi-factor matching (age, gender, smoking, duration, combinations)
- Precautionary principle

#### 5. Output Schema
A concrete JSON example with field types shown by example for reliable parsing.

### Temperature: 0.1
Clinical assessments must be deterministic. A small amount of temperature allows nuanced reasoning.

### Function Declaration Design

**`get_patient_data`** — Simple single-parameter tool. Description lists exact fields returned.

**`search_clinical_guidelines`** — Description mentions "semantic search" and "NICE NG12". Query parameter includes example queries. `top_k` is optional with default.

---

## 2. Chat Agent (`src/chat.py`)

### Location

`CHAT_SYSTEM_PROMPT` constant in `src/chat.py`, passed via `system_instruction` at model init.

### Design Goals

| Goal | How the prompt achieves it |
|---|---|
| **Evidence-only answers** | "Base every answer on the CONTEXT passages. Do NOT use general medical knowledge." |
| **Mandatory citations** | "Include an inline citation in the format [NG12 p.XX]." |
| **Graceful refusal** | Explicit refusal template: "I couldn't find support in the NG12 text for that..." |
| **No hallucination** | "Never fabricate thresholds, age cut-offs, or referral criteria." |
| **Multi-turn awareness** | "You support multi-turn conversations — the user may ask follow-up questions." |
| **Structured output** | JSON schema with answer + citations array. |

### Prompt Structure: Role → Rules → Output

The chat prompt is simpler than the assessment prompt because the chat agent doesn't use function calling — retrieval is done server-side before the prompt is built.

#### 1. Role Definition
```
You are a clinical guideline assistant specialising in the NICE NG12 guidelines.
```
"Assistant" (not "agent") because this mode is answering questions, not making clinical decisions.

#### 2. Rules (5 key constraints)
1. **Evidence-only** — use only the provided CONTEXT passages
2. **Cite sources** — inline `[NG12 p.XX]` format
3. **Refuse gracefully** — explicit template for insufficient evidence
4. **Do not invent** — no fabricated thresholds or criteria
5. **Be concise** — answer directly, then supporting detail

These rules are numbered for emphasis and to make each constraint individually clear to the model.

#### 3. Output Schema
JSON with `answer` (containing inline citations) and `citations` array (source, page, chunk_id, excerpt).

### Why no function calling?

The chat agent performs RAG retrieval server-side before calling Gemini, for three reasons:

1. **Simpler** — the user's message already IS the search query; no need for the model to formulate one
2. **Faster** — one LLM call instead of a multi-turn function-calling loop
3. **More controllable** — the server decides how many passages to retrieve (`top_k`)

### Context Assembly

The prompt injected into each chat call has three sections:

```
CONTEXT:
--- [chunk_0] Page 40 | Section title ---
(guideline text)

CONVERSATION HISTORY:
USER: previous question
ASSISTANT: previous answer

USER QUESTION: new question
```

The CONTEXT block uses chunk IDs and page numbers so the model can cite them. The HISTORY block includes the last 20 messages (10 turns) to support follow-up questions.

### Temperature: 0.2
Slightly higher than the assessment agent (0.1) because chat answers benefit from more natural language, while still being grounded in evidence.

### Refusal Behaviour

The prompt explicitly instructs the model to refuse when evidence is insufficient, with a template:

```
"I couldn't find support in the NG12 text for that. The retrieved passages cover: ..."
```

This prevents the model from defaulting to its training data when the vector store doesn't contain relevant passages.

---

## Comparison

| Aspect | Assessment Agent | Chat Agent |
|---|---|---|
| System prompt | `SYSTEM_PROMPT` in `agent.py` | `CHAT_SYSTEM_PROMPT` in `chat.py` |
| Interaction | Function calling (multi-turn) | Single generation call |
| RAG | Model decides what to search | Server retrieves before calling model |
| Output | Structured risk assessment JSON | Natural language + citations JSON |
| Temperature | 0.1 | 0.2 |
| Memory | Stateless (per-request) | Stateful (session history) |
| Refusal | N/A (always produces assessment) | Explicit refusal when evidence is insufficient |

---

## Iteration History

### Assessment prompt
1. **v1** — Basic instruction. Problem: model answered without calling tools.
2. **v2** — Added explicit workflow. Problem: single generic search query.
3. **v3** (current) — Example queries in tool descriptions, "one or more times". Model now makes targeted searches.

### Chat prompt
1. **v1** (current) — Evidence-only rules, citation format, refusal template, conversation history support. Designed based on lessons from the assessment prompt.

### Future improvements
- Few-shot examples for both prompts to improve citation quality.
- `response_mime_type="application/json"` for guaranteed JSON output.
- Chat: allow the model to request additional searches if initial retrieval is insufficient.
