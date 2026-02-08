"""Streamlit UI for Clinical Decision Support Agent.

Simple frontend to input Patient ID and view risk assessment results.
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any
import json

# Configuration
API_BASE_URL = "http://localhost:8000"  # TODO: Make this configurable via environment variable


def fetch_patients() -> Optional[list]:
    """Fetch list of available patient IDs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/patients")
        response.raise_for_status()
        data = response.json()
        return data.get("patients", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching patients: {e}")
        return None


def assess_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Send assessment request to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/assess",
            json={"patient_id": patient_id}
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Clinical Decision Support Agent",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Clinical Decision Support Agent")
    st.markdown("**Cancer Risk Assessment using NICE NG12 Guidelines**")
    st.divider()
    
    # Sidebar for patient selection
    with st.sidebar:
        st.header("Patient Selection")
        
        # Fetch and display available patients
        if st.button("Refresh Patient List"):
            st.rerun()
        
        patients = fetch_patients()
        if patients:
            st.success(f"Found {len(patients)} patients")
            selected_patient = st.selectbox(
                "Select a patient:",
                options=[""] + patients,
                format_func=lambda x: "Choose a patient..." if x == "" else x
            )
        else:
            selected_patient = None
            st.warning("Could not load patient list. Check API connection.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Assessment")
        patient_id_input = st.text_input(
            "Enter Patient ID:",
            value=selected_patient if selected_patient else "",
            placeholder="e.g., PT-101",
            help="Enter a patient ID to assess their cancer risk"
        )
    
    with col2:
        st.subheader("API Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Response Error")
        except requests.exceptions.RequestException:
            st.error("❌ API Not Available")
            st.info(f"Make sure the API is running at {API_BASE_URL}")
    
    # Assessment button and results
    if st.button("Assess Patient", type="primary", disabled=not patient_id_input):
        if not patient_id_input:
            st.warning("Please enter a Patient ID")
        else:
            with st.spinner("Assessing patient risk..."):
                result = assess_patient(patient_id_input)
                
                if result:
                    st.success("Assessment Complete!")
                    st.divider()
                    
                    # Display assessment results
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
                    
                    # Reasoning
                    st.subheader("Reasoning")
                    st.write(result["reasoning"])
                    
                    # Relevant Symptoms
                    if result.get("relevant_symptoms"):
                        st.subheader("Relevant Symptoms")
                        symptoms_list = ", ".join(result["relevant_symptoms"])
                        st.write(symptoms_list)
                    
                    # Citations
                    if result.get("citations"):
                        st.subheader("Clinical Guideline Citations")
                        for i, citation in enumerate(result["citations"], 1):
                            with st.expander(f"Citation {i}: Page {citation['page_number']} - {citation['section']}"):
                                st.write(f"**Relevance Score:** {citation['relevance_score']:.2%}")
                                st.write(f"**Content:**")
                                st.write(citation["content"])
                    
                    # Raw JSON (expandable)
                    with st.expander("View Raw JSON Response"):
                        st.json(result)
    
    # Footer
    st.divider()
    st.caption("Powered by Google Vertex AI (Gemini 1.5) | NICE NG12 Guidelines")


if __name__ == "__main__":
    main()
