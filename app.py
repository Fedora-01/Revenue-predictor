import streamlit as st
import joblib
import pandas as pd

st.title('Clasificador de Ganancias de Películas')

# ========================
# LOAD ARTIFACTS
# ========================
@st.cache_resource
def load_artifacts():
    model = joblib.load('automl_model.pkl')
    prep = joblib.load('column_transformer.pkl')
    le = joblib.load('label_encoder_Y.pkl')
    mlb_genres = joblib.load('multilabel_binarizer_genres.pkl')
    mlb_companies = joblib.load('multilabel_binarizer_companies.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    svd = joblib.load('svd.pkl')

    return model, prep, le, mlb_genres, mlb_companies, vectorizer, svd


model, prep, le, mlb_genres, mlb_companies, vectorizer, svd = load_artifacts()

# ========================
# UI
# ========================
st.header("Introduce los datos de la película")

vote_average = st.slider("vote_average", 0.0, 10.0, 7.0)
vote_count = st.number_input("vote_count", 0, 100000, 1000)
runtime = st.number_input("runtime", 0, 300, 90)
adult = st.checkbox("adult")
budget = st.number_input("budget", 0, 1_000_000_000, 1_000_000)
popularity = st.number_input("popularity", 0.0, 1000.0, 10.0)

status = st.text_input("status", "Released")
original_language = st.text_input("original_language", "en")

genres_input = st.text_input("genres (lista separada por coma)", "Action,Comedy")
companies_input = st.text_input("companies (lista separada por coma)", "Disney")
overview_text = st.text_area("overview", "A movie story")

# ========================
# PREDICT
# ========================
if st.button("Predecir"):

    # Convertir inputs
    genres_list = [g.strip() for g in genres_input.split(",") if g.strip()]
    companies_list = [c.strip() for c in companies_input.split(",") if c.strip()]

    # Dataset base (CRUDO)
    data = {
        'title': ['unknown'],
        'vote_average': [vote_average],
        'vote_count': [vote_count],
        'status': [status],
        'release_date': ['unknown'],
        'runtime': [runtime],
        'adult': [adult],
        'backdrop_path': ['unknown'],
        'budget': [budget],
        'homepage': ['unknown'],
        'imdb_id': ['unknown'],
        'original_language': [original_language],
        'original_title': ['unknown'],
        'overview': [overview_text],
        'popularity': [popularity],
        'poster_path': ['unknown'],
        'tagline': ['unknown'],
        'genres': [genres_list],
        'production_companies': [companies_list],
        'production_countries': ['unknown'],
        'spoken_languages': ['unknown']
    }

    df = pd.DataFrame(data)

    # ========================
    # TRANSFORM (ÚNICO PIPELINE)
    # ========================
    X = prep.transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    final_input = pd.DataFrame(X)

    # ========================
    # PREDICT
    # ========================
    pred = model.predict(final_input)
    label = le.inverse_transform(pred)

    st.success(f"Resultado: {label[0]}")
