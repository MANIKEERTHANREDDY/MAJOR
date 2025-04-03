import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json
import re

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key is None:
    raise ValueError("No API key found in environment variables")

genai.configure(api_key=api_key)

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return None
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None

def analyze_text(text):
    prompt = f"""
    Extract and return only the following biomedical entities from the given text:
    - Diseases
    - Drugs
    - Genes/Proteins
    - Symptoms
    
    Present the results in a structured JSON format with these categories:
    {{
        "diseases": [],
        "drugs": [],
        "genes_proteins": [],
        "symptoms": []
    }}

    For each entity, include:
    - The exact text found
    - The sentence or phrase where it appears (context)
    
    Ensure the response is valid JSON format.
    
    Text to analyze:
    {text}
    """

    result = get_gemini_response(prompt)
    
    if result:
        try:
            json_text = extract_json(result)
            if json_text:
                parsed_result = json.loads(json_text)
                entity_list = []
                diseases = []
                for category, entities in parsed_result.items():
                    for entity in entities:
                        if isinstance(entity, dict) and "text" in entity and "context" in entity:
                            entity_list.append([category.capitalize(), entity["text"], entity["context"]])
                            if category == "diseases":
                                diseases.append(entity["text"])
                return entity_list, diseases
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            st.write("Raw Response:")
            st.code(result)
            return None, []
    return None, []

def get_drugs_for_diseases(diseases):
    if not diseases:
        return {}
    
    prompt = f"""
    For each of the following diseases, recommend the top 3 most commonly used drugs names only.
    
    Return the results in JSON format:
    {{
        "disease_name": ["drug1", "drug2", "drug3"]
    }}
    
    Diseases:
    {diseases}
    """
    
    result = get_gemini_response(prompt)
    
    if result:
        try:
            json_text = extract_json(result)
            if json_text:
                return json.loads(json_text)
        except json.JSONDecodeError:
            return {}
    return {}

def process_file(file):
    if file.name.endswith('.txt'):
        text = file.getvalue().decode('utf-8')
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
        text = ' '.join(df.astype(str).values.flatten())
    else:
        st.error("Unsupported file format. Please upload a .txt or .csv file.")
        return None, []
    
    return analyze_text(text)

def display_results(results):
    if results:
        df = pd.DataFrame(results, columns=["Category", "Entity", "Context"])
        st.dataframe(df, use_container_width=True)

def display_recommendations(drug_recommendations):
    if drug_recommendations:
        st.subheader("Drug Recommendations")
        for disease, drugs in drug_recommendations.items():
            st.write(f"**{disease}**: {', '.join(drugs)}")

def main():
    st.set_page_config(
        page_title="Biomedical Entity Recognition & Drug Recommendation",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("Biomedical Entity Recognition & Drug Recommendation")
    st.write("This application identifies biomedical entities")

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        st.session_state.diseases = []
        st.session_state.drug_recommendations = {}

    tab1, tab2 = st.tabs(["Text Input", "File Upload"])

    with tab1:
        st.header("Enter Text")
        text_input = st.text_area("Enter your biomedical text here:", height=200)
        if st.button("Analyze Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing text... This may take a few moments."):
                    result, diseases = analyze_text(text_input)
                    if result:
                        st.session_state.analysis_results = result
                        st.session_state.diseases = diseases
                        st.session_state.drug_recommendations = {}
                        st.subheader("Analysis Results")
                        display_results(result)
    
    with tab2:
        st.header("Upload File")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if st.button("Analyze File", type="primary"):
                with st.spinner("Analyzing file... This may take a few moments."):
                    result, diseases = process_file(uploaded_file)
                    if result:
                        st.session_state.analysis_results = result
                        st.session_state.diseases = diseases
                        st.session_state.drug_recommendations = {}
                        st.subheader("Analysis Results")
                        display_results(result)
    
    if st.session_state.analysis_results:
        if st.button("Get Drug Recommendations"):
            with st.spinner("Fetching drug recommendations..."):
                st.session_state.drug_recommendations = get_drugs_for_diseases(st.session_state.diseases)
            display_recommendations(st.session_state.drug_recommendations)

if __name__ == "__main__":
    main()
