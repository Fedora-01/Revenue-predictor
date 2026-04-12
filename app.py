import streamlit as st
import joblib
import pandas as pd

st.title("Clasificador de películas")

# =========================
# LOAD MODELO
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

pipeline = load_model()

# =========================
# INPUTS
# =========================
vote_average = st.slider("vote_average", 0.0, 10.0, 7.0)
vote_count = st.number_input("vote_count", 0, 100000, 1000)
runtime = st.number_input("runtime", 0, 300, 90)
budget = st.number_input("budget", 0, 1_000_000_000, 1_000_000)
popularity = st.number_input("popularity", 0.0, 1000.0, 10.0)

status = st.text_input("status", "Released")
original_language = st.text_input("original_language", "en")
overview = st.text_area("overview", "A movie story")

# =========================
# PREDICT
# =========================
if st.button("Predecir"):

    df = pd.DataFrame([{
        "vote_average": vote_average,
        "vote_count": vote_count,
        "runtime": runtime,
        "budget": budget,
        "popularity": popularity,
        "status": status,
        "original_language": original_language,
        "overview": overview
    }])

    pred = pipeline.predict(df)

    st.success(f"Resultado: {pred[0]}")
