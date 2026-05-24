import streamlit as st
import tempfile
import os
import joblib
import PyPDF2
import re

model = joblib.load("classifier_fr.pkl")
vectorizer = joblib.load("vectorizer_fr.pkl")
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'[^a-zA-ZÀ-ÿ ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


def extract_text_from_pdf(file_path):

    text = ""

    with open(file_path, "rb") as file:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + " "

    return text


def predict_cv(text):

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    scores = model.decision_function(vectorized_text)

    confidence = round(scores.max() * 10, 2)

    return prediction, confidence


st.set_page_config(
    page_title="AI CV Classification",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI CV Classification System")

st.write(
    "Upload a CV in PDF format and the AI model "
    "will automatically predict its category."
)

uploaded_file = st.file_uploader(
    "Upload a CV PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("CV uploaded successfully!")

    st.write("### 📁 File Name")
    st.write(uploaded_file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    text = extract_text_from_pdf(temp_path)

    prediction, confidence = predict_cv(text)

    st.write("## 🎯 Prediction Result")

    st.info(f"Predicted Category: {prediction}")

    st.write("### Confidence Score")

    progress_value = min(int(confidence), 100)

    st.progress(progress_value)

    st.write(f"{confidence}%")

    st.write("### 📄 Extracted Text Preview")

    st.text_area(
        "CV Content",
        text[:1500],
        height=300
    )

    os.remove(temp_path)