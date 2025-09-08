import streamlit as st
import joblib
import pandas as pd
from io import StringIO
import PyPDF2

st.set_page_config(page_title="AI Resume Screening", page_icon="🧠", layout="wide")
st.title("AI-based Resume Screening System")
st.caption("Upload a resume to get the best-fit job category")

@st.cache_resource
def load_model():
    return joblib.load("model/tfidf_logreg.joblib")

model = load_model()

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

with st.sidebar:
    st.header("About")
    st.markdown("This app uses TF-IDF + Logistic Regression to classify resumes into categories.")
    st.markdown("**Categories:** Data Scientist, Software Engineer, HR Specialist, Marketing Analyst, Project Manager")

tab1, tab2 = st.tabs(["Predict", "Batch Predict"])

with tab1:
    uploaded = st.file_uploader("Upload resume", type=["txt", "pdf"])
    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            content = read_pdf(uploaded)
        else:
            content = uploaded.read().decode("utf-8", errors="ignore")
        if st.button("Predict"):
            pred = model.predict([content])[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([content])[0].max()
            st.success(f"Prediction: **{pred}**")
            if proba is not None:
                st.write(f"Confidence: **{proba:.2f}**")
            st.text_area("Extracted text", content[:3000], height=200)

with tab2:
    st.write("Upload a CSV file with a column named **text**")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if csv_file:
        df = pd.read_csv(csv_file)
        preds = model.predict(df["text"])
        out = df.copy()
        out["prediction"] = preds
        st.dataframe(out.head(20))
        st.download_button("Download predictions", out.to_csv(index=False), file_name="predictions.csv")