"""Streamlit UI for Clinical Decision Support Agent.

Two tabs:
1. **Assessment** — Input a Patient ID, get a risk assessment with citations.
2. **Chat** — Ask questions about the NICE NG12 guidelines in a multi-turn conversation.
"""

import uuid

import requests
import streamlit as st
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_patients() -> Optional[list]:
    """Fetch list of available patient IDs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/patients", timeout=5)
        response.raise_for_status()
        return response.json().get("patients", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patients: {e}")
        return None


def assess_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Send assessment request to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/assess",
            json={"patient_id": patient_id},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"Patient ID '{patient_id}' not found.")
        else:
            st.error(f"Error assessing patient: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None


def send_chat_message(
    session_id: str, message: str, top_k: int = 5
) -> Optional[Dict[str, Any]]:
    """Send a chat message to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": session_id, "message": message, "top_k": top_k},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def fetch_chat_history(session_id: str) -> Optional[list]:
    """Fetch conversation history from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/{session_id}/history", timeout=5
        )
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.exceptions.RequestException:
        return None


def delete_chat_session(session_id: str) -> bool:
    """Delete a chat session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat/{session_id}", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Decision Support Agent",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical Decision Support Agent")
st.markdown("**NICE NG12 Cancer Guidelines — Assessment & Chat**")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — shared
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("API Status")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.success("API Connected")
        else:
            st.warning("API Response Error")
    except requests.exceptions.RequestException:
        st.error("API Not Available")
        st.info(f"Make sure the API is running at {API_BASE_URL}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_assess, tab_chat = st.tabs(["Assessment", "Chat"])


# ===== TAB 1 — ASSESSMENT =================================================

with tab_assess:
    with st.sidebar:
        st.header("Patient Selection")
        if st.button("Refresh Patient List"):
            st.rerun()
        patients = fetch_patients()
        if patients:
            st.success(f"Found {len(patients)} patients")
            selected_patient = st.selectbox(
                "Select a patient:",
                options=[""] + patients,
                format_func=lambda x: "Choose a patient..." if x == "" else x,
            )
        else:
            selected_patient = None
            st.warning("Could not load patient list.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Patient Assessment")
        patient_id_input = st.text_input(
            "Enter Patient ID:",
            value=selected_patient if selected_patient else "",
            placeholder="e.g., PT-101",
            help="Enter a patient ID to assess their cancer risk",
        )
    with col2:
        st.write("")  # spacer

    if st.button("Assess Patient", type="primary", disabled=not patient_id_input):
        if not patient_id_input:
            st.warning("Please enter a Patient ID")
        else:
            with st.spinner("Assessing patient risk..."):
                result = assess_patient(patient_id_input)
                if result:
                    st.success("Assessment Complete!")
                    st.divider()

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Patient ID", result["patient_id"])
                    with col_b:
                        assessment = result["assessment"]
                        if assessment == "Urgent Referral":
                            st.error(f"🔴 {assessment}")
                        elif assessment == "Urgent Investigation":
                            st.warning(f"🟡 {assessment}")
                        else:
                            st.info(f"🟢 {assessment}")
                    with col_c:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                    st.subheader("Reasoning")
                    st.write(result["reasoning"])

                    if result.get("relevant_symptoms"):
                        st.subheader("Relevant Symptoms")
                        st.write(", ".join(result["relevant_symptoms"]))

                    if result.get("citations"):
                        st.subheader("Clinical Guideline Citations")
                        for i, cit in enumerate(result["citations"], 1):
                            with st.expander(
                                f"Citation {i}: Page {cit['page_number']} — {cit['section']}"
                            ):
                                st.write(f"**Relevance Score:** {cit['relevance_score']:.2%}")
                                st.write(cit["content"])

                    with st.expander("View Raw JSON Response"):
                        st.json(result)


# ===== TAB 2 — CHAT =======================================================

with tab_chat:
    # -- session management in sidebar
    with st.sidebar:
        st.header("Chat Session")
        if "chat_session_id" not in st.session_state:
            st.session_state.chat_session_id = str(uuid.uuid4())[:8]
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        st.text_input(
            "Session ID",
            value=st.session_state.chat_session_id,
            key="sidebar_session_id",
            disabled=True,
        )

        if st.button("New Session"):
            st.session_state.chat_session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_messages = []
            st.rerun()

        if st.button("Clear Chat"):
            delete_chat_session(st.session_state.chat_session_id)
            st.session_state.chat_messages = []
            st.rerun()

    st.subheader("Ask about NG12 Guidelines")
    st.caption(
        "Ask questions like: *What symptoms trigger an urgent referral for lung cancer?* "
        "or *Does persistent hoarseness require urgent referral?*"
    )

    # -- display existing messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("Citations"):
                    for cit in msg["citations"]:
                        page = cit.get("page", "?")
                        excerpt = cit.get("excerpt", "")
                        st.caption(f"**[NG12 p.{page}]** {excerpt[:300]}")

    # -- chat input
    if user_input := st.chat_input("Type your question about NG12 guidelines..."):
        # Show user message immediately
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Searching guidelines..."):
                result = send_chat_message(
                    session_id=st.session_state.chat_session_id,
                    message=user_input,
                )

            if result:
                answer = result.get("answer", "Sorry, I could not generate an answer.")
                citations = result.get("citations", [])

                st.markdown(answer)
                if citations:
                    with st.expander("Citations"):
                        for cit in citations:
                            page = cit.get("page", "?")
                            excerpt = cit.get("excerpt", "")
                            st.caption(f"**[NG12 p.{page}]** {excerpt[:300]}")

                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )
            else:
                err = "Sorry, something went wrong. Check the API connection."
                st.error(err)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": err, "citations": []}
                )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Powered by Google Vertex AI (Gemini) | NICE NG12 Guidelines")
