import streamlit as st
import joblib
import pandas as pd

st.title('Clasificador de Ganancias de Películas de Animación')


@st.cache_resource
def load_artifacts():
    automl_model = joblib.load('automl_model.pkl')
    prep = joblib.load('column_transformer.pkl')
    LE = joblib.load('label_encoder_Y.pkl')
    mlb_genres = joblib.load('multilabel_binarizer_genres.pkl')
    mlb_companies = joblib.load('multilabel_binarizer_companies.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    svd = joblib.load('svd.pkl')
    
    
    feature_columns = joblib.load('feature_columns.pkl')

    return automl_model, prep, LE, mlb_genres, mlb_companies, vectorizer, svd, feature_columns


automl_model, prep, LE, mlb_genres, mlb_companies, vectorizer, svd, feature_columns = load_artifacts()


status_categories = prep.named_transformers_['N_num'].categories_[0].tolist()
language_categories = prep.named_transformers_['N_num'].categories_[1].tolist()
all_possible_genres = mlb_genres.classes_.tolist()
all_possible_companies = mlb_companies.classes_.tolist()


st.header('Introduce los detalles de la película:')

vote_average = st.slider('vote_average', 0.0, 10.0, 7.0)
vote_count = st.number_input('vote_count', min_value=0, value=1000)
runtime = st.number_input('runtime', min_value=0, value=90)
adult = st.checkbox('adult')
budget = st.number_input('budget', min_value=0, value=1000000)
popularity = st.number_input('popularity', min_value=0.0, value=10.0)

status = st.selectbox('status', status_categories)
original_language = st.selectbox('original_language', language_categories)

genres_input = st.multiselect('genres', all_possible_genres)
companies_input = st.multiselect('production_companies', all_possible_companies)

overview_text = st.text_area('overview', 'A brief description of the movie.')


if st.button('Predecir Categoría de Ganancia'):


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
        'genres': [genres_input],  
        'production_companies': [companies_input],  
        'production_countries': ['unknown'],
        'spoken_languages': ['unknown']
    }

    df = pd.DataFrame(data)

    df = df[prep.feature_names_in_]
    X_prep = prep.transform(df)

    if hasattr(X_prep, "toarray"):
        X_prep = X_prep.toarray()

    processed_df = pd.DataFrame(X_prep)

    genres_encoded = mlb_genres.transform([genres_input])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_)

    companies_encoded = mlb_companies.transform([companies_input])
    companies_df = pd.DataFrame(companies_encoded, columns=mlb_companies.classes_)

    overview_tfidf = vectorizer.transform([overview_text])
    overview_svd = svd.transform(overview_tfidf)

    overview_cols = [f"svd_{i}" for i in range(overview_svd.shape[1])]
    overview_df = pd.DataFrame(overview_svd, columns=overview_cols)

    processed_df.index = [0]
    genres_df.index = [0]
    companies_df.index = [0]
    overview_df.index = [0]

    final_input_df = pd.concat(
        [processed_df, genres_df, companies_df, overview_df],
        axis=1
    )

    final_input_df = final_input_df.reindex(columns=feature_columns, fill_value=0)

    prediction_encoded = automl_model.predict(final_input_df)
    prediction = LE.inverse_transform(prediction_encoded)

    st.success(f'Categoría predicha: **{prediction[0]}**')

    with st.expander("Debug info"):
        st.write("Shape:", final_input_df.shape)
        st.write(final_input_df.head())
