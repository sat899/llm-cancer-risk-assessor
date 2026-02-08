# Prompt Engineering Strategy

This document explains the system prompt design for the Clinical Decision Support Agent.

## System Prompt Location

The system prompt lives in `src/agent.py` as the `SYSTEM_PROMPT` constant and is passed to Gemini via `system_instruction` at model initialisation time.

## Design Goals

| Goal | How the prompt achieves it |
|---|---|
| **Grounded in evidence** | The agent is explicitly told to base assessments on NG12 text returned by `search_clinical_guidelines`, not general medical knowledge. |
| **Structured output** | A strict JSON schema is defined in the prompt so the response can be parsed programmatically. |
| **Transparent reasoning** | The `reasoning` field forces the model to explain its clinical logic, making assessments auditable. |
| **Citation accuracy** | The prompt requires page numbers and section titles from the guideline passages. |
| **Safety-first** | The "when in doubt, choose higher urgency" rule encodes the precautionary principle. |

## Prompt Structure

The prompt follows a **Role → Workflow → Rules → Output** pattern:

### 1. Role Definition

```
You are a clinical decision support agent specialising in cancer risk
assessment using the NICE NG12 guidelines...
```

A concise identity statement. By naming the specific guideline (NG12) and the task (risk assessment), we prime the model to stay in domain and avoid hallucinating advice from other frameworks.

### 2. Workflow (Step-by-step)

```
1. RETRIEVE — Call get_patient_data(patient_id) ...
2. SEARCH  — Call search_clinical_guidelines(query) ...
3. ANALYSE — Compare the patient's presentation against ...
4. ASSESS  — Determine the appropriate risk category.
```

Explicit ordering is critical for function-calling agents. Without it, the model may attempt to answer before gathering data. Each step names the exact tool to call and what to do with the result.

**Why multiple searches?** The prompt says "one or more times" because a patient with multiple symptoms (e.g. hemoptysis + fatigue) may match guidelines in different sections. Allowing repeated search calls lets the agent cover all relevant pathways.

### 3. Assessment Categories

Three tiers are defined with clear clinical meanings:

- **Urgent Referral** — maps to the NG12 "suspected cancer pathway referral" (2-week wait)
- **Urgent Investigation** — further tests needed but not full referral criteria
- **Routine** — no urgent action required per NG12

These match the categories used in the NG12 document itself, so the model can directly map guideline language to output categories.

### 4. Rules

Key constraints:

- **Evidence-only reasoning** — prevents the model from using training-data medical knowledge that may conflict with or go beyond NG12.
- **Multi-factor matching** — instructs the model to consider age, gender, smoking history, duration, and symptom combinations (all factors the NG12 uses for its criteria).
- **Precautionary principle** — in ambiguous cases, escalate rather than downplay.

### 5. Output Schema

A concrete JSON example is provided. This is more reliable than describing the schema in prose because the model can pattern-match against it. The field types (int, float, string, array) are shown by example.

## Temperature Choice

`temperature=0.1` is used because:

- Clinical assessments must be **deterministic and reproducible**.
- We want the model to follow the structured output format reliably.
- A small amount of temperature (vs 0.0) allows the model to express nuanced reasoning without being overly rigid.

## Function Declaration Design

The two tool declarations are designed to give the model enough context to use them correctly:

### `get_patient_data`
- Simple single-parameter tool.
- Description mentions the exact fields returned (age, gender, symptoms, etc.) so the model knows what data it will get.

### `search_clinical_guidelines`
- The description mentions "semantic search" and "NICE NG12" to orient the model.
- The `query` parameter description includes example queries to demonstrate the expected format: combining symptoms with patient characteristics.
- `top_k` is optional with a default, so the model can request more results for complex cases.

## Iteration & Refinement

This prompt was developed iteratively:

1. **v1** — Basic instruction to assess patients. Problem: model would answer without calling tools.
2. **v2** — Added explicit workflow steps. Problem: model would make one generic search query.
3. **v3** (current) — Added example queries in tool descriptions and "one or more times" language. The model now makes targeted, symptom-specific searches and considers multiple cancer pathways.

## Future Improvements (Phase 2)

- Add a **chat mode** system prompt variant that maintains conversation context for follow-up questions.
- Add few-shot examples of correct assessments to improve citation quality.
- Consider using `response_mime_type="application/json"` with a formal Gemini response schema for guaranteed JSON output.
