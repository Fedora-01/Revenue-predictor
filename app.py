import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import ast # Importar ast para poder parsear los Json como strings
import numpy as np

st.title('Clasificador de Ganancias de Películas de Animación')

# Cargar el modelo y los preprocesadores
@st.cache_resource #Esto sirve para usar cosas durante la ejecucion (no se puede almacenar en un database)
def load_artifacts():
    automl_model = joblib.load('automl_model.pkl')
    column_transformer = joblib.load('column_transformer.pkl')
    label_encoder_y = joblib.load('label_encoder_Y.pkl')
    mlb_genres = joblib.load('multilabel_binarizer_genres.pkl')
    mlb_companies = joblib.load('multilabel_binarizer_companies.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    svd = joblib.load('svd.pkl')

    return automl_model, column_transformer, label_encoder_y, mlb_genres, mlb_companies, vectorizer, svd

model_feature_names = joblib.load('model_feature_names.pkl')
automl_model, prep, LE, mlb_genres, mlb_companies, vectorizer, svd = load_artifacts()

#Columnas usadas durante el entrenamiento
XC_names = ['vote_average', 'vote_count', 'runtime', 'adult', 'budget', 'popularity']
XO_names_original = ['status', 'original_language']

# Opciones para las características categóricas
status_categories = prep.named_transformers_['N_num'].categories_[0].tolist()
language_categories = prep.named_transformers_['N_num'].categories_[1].tolist()
all_possible_genres = mlb_genres.classes_.tolist()
all_possible_companies = mlb_companies.classes_.tolist()

#Interfaz
st.header('Introduce los detalles de la película:')

vote_average = st.slider('Puntuación promedio de votos (vote_average)', 0.0, 10.0, 7.0)
vote_count = st.number_input('Número de votos (vote_count)', min_value=0, value=1000)
runtime = st.number_input('Duración en minutos (runtime)', min_value=0, value=90)
adult = st.checkbox('¿Es una película para adultos? (adult)')
budget = st.number_input('Presupuesto (budget)', min_value=0, value=1000000)
popularity = st.number_input('Popularidad (popularity)', min_value=0.0, value=10.0)

status = st.selectbox('Estado de producción (status)', status_categories)
origen_language = st.selectbox('Idioma original (original_language)', language_categories)
genres_input = st.multiselect('Géneros (genres)', all_possible_genres)
production_companies_input = st.multiselect('Compañías de producción (production_companies)', all_possible_companies)
overview_text = st.text_area('Resumen de la película (overview)', 'A brief description of the movie.')

# Entrada de datos y prediccion
if st.button('Predecir Categoría de Ganancia'):
    # Creando un dataframe con los datos ingresados
    data = {
        'title': ['unkown'],
        'vote_average': [vote_average],
        'vote_count': [vote_count],
        'status': [status],
        'release_date': ['unkown'],
        'runtime': [runtime],
        'adult': [adult],
        'backdrop_path': ['unkown'],
        'budget': [budget],
        'homepage': ['unkown'],
        'imdb_id': ['unkown'],
        'original_language': [origen_language],
        'original_title': ['unkown'],
        'overview': ['unkown'],
        'popularity': [popularity],
        'poster_path': ['unkown'],
        'tagline': ['unkown'],
        'genres': ['unkown'],
        'production_companies': ['unkown'],
        'production_countries': ['unkown'],
        'spoken_languages': ['unkown']
    }

    #Creacion de un dataframe con datos de entrada
    df_prep_transform = pd.DataFrame(data)
    df_prep_transform = df_prep_transform[prep.feature_names_in_]

    p_features_array = prep.transform(df_prep_transform).toarray()

    # Obtiene el arreglo de los nombres de las columnas
    ohe_feature_names = prep.named_transformers_['N_num'].get_feature_names_out(XO_names_original)
    initial_processed_columns = XC_names + list(ohe_feature_names)
    processed_df = pd.DataFrame(p_features_array, columns=initial_processed_columns)

    #Se procesan los generos con mlb
    genre_encoded_input = mlb_genres.transform([genres_input])
    genre_df_input = pd.DataFrame(genre_encoded_input, columns=mlb_genres.classes_)

    #Se procesan las casas productoras
    companies_encoded_input = mlb_companies.transform([production_companies_input])
    companies_df_input = pd.DataFrame(companies_encoded_input, columns=mlb_companies.classes_)

    # Se procesa la reseña con tfid y SVD
    overview_tfidf = vectorizer.transform([overview_text])
    overview_svd_reduced = svd.transform(overview_tfidf)

    # Crear un dataframe con exactamente 100 columnas
    overview_svd_array = overview_svd_reduced[0]
    if len(overview_svd_array) < 100:
        overview_svd_array = np.pad(overview_svd_array, (0, 100 - len(overview_svd_array)))
    else:
        overview_svd_array = overview_svd_array[:100]

    overview_df_input = pd.DataFrame([overview_svd_array], columns=[str(i) for i in range(100)])

    # Establecer índices
    processed_df.index = [0]
    genre_df_input.index = [0]
    companies_df_input.index = [0]
    overview_df_input.index = [0]

    # Concatenar todos los dataframes
    final_input_df = pd.concat([processed_df, genre_df_input, companies_df_input, overview_df_input], axis=1)

    # Asegurar que tenga exactamente las mismas columnas
    for col in model_feature_names:
        if col not in final_input_df.columns:
            final_input_df[col] = 0

    # Seleccionar solo las columnas que el modelo espera, en el orden correcto
    final_input_df = final_input_df[model_feature_names]

    # Realizar la predicción
    prediction_encoded = automl_model.predict(final_input_df)
    prediction_label = LE.inverse_transform(prediction_encoded)

    st.success(f'La categoría de ganancia predicha es: **{prediction_label[0]}**')
