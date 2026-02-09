"""Agent logic for clinical decision support using Gemini 1.5.

This module implements the core reasoning engine:
1. Receives a patient ID
2. Uses function calling to retrieve patient data (tool)
3. Uses function calling to search NG12 guidelines via RAG (tool)
4. Synthesises data + guidelines to produce a risk assessment
5. Returns structured JSON with citations

The agent never accesses data directly — it can only act through the
tool functions declared in ``src.tools``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import config
from src.tools import get_patient_data, search_clinical_guidelines

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt  (see PROMPTS.md for design rationale)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical decision support agent specialising in cancer risk \
assessment using the NICE NG12 guidelines ("Suspected cancer: recognition \
and referral").

## Your Role
Assess whether a patient should receive an **Urgent Referral**, \
**Urgent Investigation**, or **Routine** follow-up based on their clinical \
presentation and the NG12 guideline criteria.

## Workflow
1. **RETRIEVE** — Call `get_patient_data(patient_id)` to obtain the \
patient's demographics, symptoms, and history.
2. **SEARCH** — Call `search_clinical_guidelines(query)` one or more times \
with targeted queries derived from the patient's symptoms, age, gender, and \
risk factors (e.g. "hemoptysis referral criteria", "breast lump urgent \
referral female over 30").
3. **ANALYSE** — Compare the patient's presentation against the retrieved \
guideline criteria.  Pay attention to age thresholds, symptom duration, \
smoking history, and symptom combinations.
4. **ASSESS** — Determine the appropriate risk category.

## Assessment Categories
- **Urgent Referral** — the patient's symptoms match NG12 criteria for a \
suspected cancer pathway referral (typically a 2-week-wait referral).
- **Urgent Investigation** — the patient's symptoms warrant urgent \
investigation (imaging, blood tests, etc.) but do not meet full referral \
criteria.
- **Routine** — the patient's symptoms do not meet any urgent criteria \
per NG12.

## Rules
- ALWAYS base your assessment on the actual NG12 text returned by \
`search_clinical_guidelines`.  Do NOT rely on general medical knowledge.
- Cite specific page numbers and guideline sections.
- Consider age, gender, smoking history, symptom duration, and symptom \
combinations when matching criteria.
- If multiple symptoms point to different cancer types, assess each pathway.
- When in doubt between categories, choose the **more cautious** (higher \
urgency) option.

## Output Format
After completing your tool calls, return **only** a JSON object \
(no markdown fences, no surrounding text):

{
    "assessment": "Urgent Referral | Urgent Investigation | Routine",
    "reasoning": "Detailed clinical reasoning explaining how the patient's \
symptoms match or do not match NG12 criteria.",
    "citations": [
        {
            "page_number": 0,
            "section": "guideline section title",
            "content": "relevant excerpt from the guideline",
            "relevance_score": 0.0
        }
    ],
    "relevant_symptoms": ["symptom1", "symptom2"],
    "confidence": 0.0
}
"""

# ---------------------------------------------------------------------------
# Gemini function declarations (tool schema)
# ---------------------------------------------------------------------------


def _build_tools():
    """Build Vertex AI Tool declarations (imported lazily)."""
    from vertexai.generative_models import FunctionDeclaration, Tool

    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_patient_data",
                description=(
                    "Retrieve structured patient data (age, gender, smoking "
                    "history, symptoms, symptom duration) from the patient "
                    "database by their unique ID."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Unique patient identifier, e.g. 'PT-101'",
                        }
                    },
                    "required": ["patient_id"],
                },
            ),
            FunctionDeclaration(
                name="search_clinical_guidelines",
                description=(
                    "Semantic search over the NICE NG12 cancer guidelines. "
                    "Returns the most relevant guideline passages with page "
                    "numbers.  Use targeted queries that combine symptoms "
                    "with patient characteristics for best results."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language query, e.g. 'unexplained "
                                "hemoptysis urgent referral criteria male "
                                "smoker over 40'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    )


# Map tool names → Python callables
_TOOL_DISPATCH = {
    "get_patient_data": get_patient_data,
    "search_clinical_guidelines": search_clinical_guidelines,
}

MAX_TOOL_ROUNDS = 10  # safety limit on iterative function calling


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class AssessmentAgent:
    """Gemini agent that assesses cancer risk via function calling."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.VERTEX_AI_MODEL_NAME
        self._model = None

    # -- model initialisation ------------------------------------------------

    def _initialize_model(self):
        """Lazy-init the GenerativeModel with tools and system prompt."""
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
            tools=[_build_tools()],
            system_instruction=SYSTEM_PROMPT,
            generation_config=GenerationConfig(
                temperature=0.1,  # low temperature for clinical accuracy
                max_output_tokens=4096,
            ),
        )
        logger.info("Initialised Gemini model: %s", self.model_name)
        return self._model

    # -- main entry point ----------------------------------------------------

    def assess(self, patient_id: str) -> Dict[str, Any]:
        """Assess cancer risk for *patient_id*.

        Orchestrates a multi-turn conversation with Gemini where the model
        decides which tools to call and when, then returns a structured
        assessment once it has enough information.

        Returns:
            Dict matching the AssessmentResponse schema.

        Raises:
            ValueError: patient not found.
            RuntimeError: agent reasoning failed.
        """
        model = self._initialize_model()
        chat = model.start_chat()

        # Kick off the conversation
        user_msg = (
            f"Assess the cancer risk for patient {patient_id}. "
            f"Start by retrieving their data, then search the NG12 "
            f"guidelines based on their symptoms, and provide your "
            f"structured JSON assessment."
        )

        response = chat.send_message(user_msg)

        # Iterative function-calling loop
        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            fn_calls = self._extract_function_calls(response)
            if not fn_calls:
                break  # model returned its final text answer

            fn_response_parts = self._execute_tool_calls(fn_calls, round_num)
            response = chat.send_message(fn_response_parts)

        return self._parse_response(patient_id, response)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_function_calls(response) -> List[tuple]:
        """Pull (name, args_dict) pairs from the model response."""
        calls = []
        for part in response.candidates[0].content.parts:
            fc = part.function_call
            if fc and fc.name:
                # fc.args is a proto Struct — convert to plain dict
                args: Dict[str, Any] = {}
                if fc.args:
                    for key, value in fc.args.items():
                        # proto numbers arrive as float; coerce int params
                        if key == "top_k" and isinstance(value, float):
                            value = int(value)
                        args[key] = value
                calls.append((fc.name, args))
        return calls

    @staticmethod
    def _execute_tool_calls(
        fn_calls: List[tuple], round_num: int
    ) -> list:
        """Execute tool functions and wrap results as Part responses."""
        from vertexai.generative_models import Part

        parts = []
        for name, args in fn_calls:
            logger.info("  ↳ round %d  tool=%s  args=%s", round_num, name, args)
            tool_fn = _TOOL_DISPATCH.get(name)
            if tool_fn is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool_fn(**args)
                except Exception as exc:
                    logger.error("Tool %s failed: %s", name, exc)
                    result = {"error": str(exc)}

            parts.append(
                Part.from_function_response(name=name, response={"result": result})
            )
        return parts

    def _parse_response(self, patient_id: str, response) -> Dict[str, Any]:
        """Parse Gemini's final text into the structured assessment dict."""
        try:
            text = response.text

            # Strip markdown code fences if the model wrapped its JSON
            json_str = text
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_str:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0]

            parsed = json.loads(json_str.strip())

            return self._format_response(
                patient_id=patient_id,
                assessment=parsed.get("assessment", "Routine"),
                reasoning=parsed.get("reasoning", ""),
                citations=parsed.get("citations", []),
                relevant_symptoms=parsed.get("relevant_symptoms", []),
                confidence=float(parsed.get("confidence", 0.5)),
            )

        except (json.JSONDecodeError, AttributeError, IndexError) as exc:
            logger.warning("Could not parse model JSON — falling back: %s", exc)
            raw_text = getattr(response, "text", str(response))
            return self._format_response(
                patient_id=patient_id,
                assessment="Routine",
                reasoning=raw_text,
                citations=[],
                relevant_symptoms=[],
                confidence=0.3,
            )

    @staticmethod
    def _format_response(
        patient_id: str,
        assessment: str,
        reasoning: str,
        citations: List[Dict[str, Any]],
        relevant_symptoms: List[str],
        confidence: float,
    ) -> Dict[str, Any]:
        """Normalise and return the final response dictionary."""
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return {
            "patient_id": patient_id,
            "assessment": assessment,
            "reasoning": reasoning,
            "citations": citations,
            "relevant_symptoms": relevant_symptoms,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
