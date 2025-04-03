import streamlit as st
from transformers import pipeline
import PyPDF2
import drug
from docx import Document
import google.generativeai as genai 
import os# Google Gemini API
from dotenv import load_dotenv
import subprocess
import sys
load_dotenv()

# Configure Google Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key is None:
    raise ValueError("No API key found in environment variables")

genai.configure(api_key=api_key)

# Load the BioBERT NER model
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model=r"D:\mk\biobert_ner_bc5cdr_jnlpba", aggregation_strategy="simple")

ner_pipeline = load_ner_pipeline()

# Function to get drug recommendations from Google Gemini
def get_drug_recommendation(disease):
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Modify the prompt to ensure a structured response
    response = model.generate_content(
        f"List only 1 to 3 drug names used to treat {disease}, separated by commas. No explanations, just drug names."
    )
    
    return response.text.strip() # Gemini returns text with drug recommendations

# Function to extract text from files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""
def  main():
    def run_check():
        python_executable = sys.executable

# Run Streamlit using the full path
        subprocess.run([python_executable, "-m", "streamlit", "run", "check.py"])
        
    run_check()
    extract_text_from_file()
    get_drug_recommendation()

# Streamlit UI
    st.title("Biomedical NER + Drug Recommendation with Gemini")

    # Text input
    text = st.text_area("Enter text for NER:")

    # File uploader
    uploaded_file = st.file_uploader("Upload a text, PDF, or Word file", type=["txt", "pdf", "docx"])

    if st.button("Analyze"):
        if uploaded_file is not None:
            text = extract_text_from_file(uploaded_file)
        
        if text.strip():
            # Get named entities from BioBERT NER model
            results = ner_pipeline(text)

            # Extract diseases from NER results
    # Extract diseases from NER results
        recognized_diseases = set()
        entities = []

        for entity in results:
            word = entity["word"]
            entity_type = entity.get("entity", entity.get("entity_group", "Unknown"))

            # Fix: Correctly detect diseases
            if "disease" in entity_type.lower() or "disorder" in entity_type.lower():
                recognized_diseases.add(word)

            entities.append({"Word": word, "Entity": entity_type})


            # Display recognized entities
            if entities:
                st.subheader("Named Entities Detected:")
                st.table(entities)
            else:
                st.info("No biomedical entities detected.")

            # Get drug recommendations for recognized diseases
            # Get drug recommendations for recognized diseases
            if recognized_diseases:
                st.subheader("Recommended Drugs:")
                
                for disease in recognized_diseases:
                    drugs = get_drug_recommendation(disease)
                    
                    if drugs:  # Ensure Gemini returned something
                        st.markdown(f"**{disease.title()}**: {drugs}")
                    else:
                        st.markdown(f"**{disease.title()}**: No drug recommendations found.")

            else:
                st.info("No diseases detected.")



        else:
            st.warning("Please enter text or upload a file for analysis.")
if __name__ == "__main__":
    main()
