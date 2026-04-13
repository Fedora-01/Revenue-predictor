import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import ast # Importar ast para poder parsear los Json como strings

st.title('Clasificador de Ganancias de Películas de Animación')

# Cargar el modelo y los preprocesadores
@st.cache_resource #Esto sirve para usar cosas durante la ejecucion (no se puede almacenar en un database)
def load_artifacts():
    automl_model = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/automl_model.pkl')
    column_transformer = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/column_transformer.pkl')
    label_encoder_y = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/label_encoder_Y.pkl')
    mlb_genres = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/multilabel_binarizer_genres.pkl')
    mlb_companies = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/multilabel_binarizer_companies.pkl')
    vectorizer = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/vectorizer.pkl')
    svd = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/svd.pkl') 
    model_feature_columns = joblib.load('/content/drive/MyDrive/Inteligencia Artificial/Modelo/feature_columns.pkl') # Cargar los nombres de las columnas

    return automl_model, column_transformer, label_encoder_y, mlb_genres, mlb_companies, vectorizer, svd, model_feature_columns

automl_model, prep, LE, mlb_genres, mlb_companies, vectorizer, svd, model_feature_columns = load_artifacts()

# Columnas usadas durante el entrenamiento
XC_names = ['vote_average', 'vote_count', 'runtime', 'adult', 'budget', 'popularity']
XO_names_original = ['status', 'original_language'] # Los nombres originales antes de aplicar el column transformer

# Opciones para las características categóricas (usando los preprocesadores)
status_categories = prep.named_transformers_['cat_encoded'].categories_[0].tolist() #status
language_categories = prep.named_transformers_['cat_encoded'].categories_[1].tolist() #lenguaje
all_possible_genres = mlb_genres.classes_.tolist() #genero
all_possible_companies = mlb_companies.classes_.tolist() #compañia

# Interfaz
st.header('Introduce los detalles de la película:')

vote_average = st.slider('Puntuación promedio de votos (vote_average)', 0.0, 10.0, 7.0)
vote_count = st.number_input('Número de votos (vote_count)', min_value=0, value=1000)
runtime = st.number_input('Duración en minutos (runtime)', min_value=0, value=90)
adult = st.checkbox('¿Es una película para adultos? (adult)')
budget = st.number_input('Presupuesto (budget)', min_value=0, value=1000000) # Mantener el valor de 1 millón como ejemplo.
popularity = st.number_input('Popularidad (popularity)', min_value=0.0, value=10.0)

status = st.selectbox('Estado de producción (status)', status_categories)
origen_language = st.selectbox('Idioma original (original_language)', language_categories)
genres_input = st.multiselect('Géneros (genres)', all_possible_genres)
production_companies_input = st.multiselect('Compañías de producción (production_companies)', all_possible_companies)
overview_text = st.text_area('Resumen de la película (overview)', 'A brief description of the movie.')

# Entrada de datos y prediccion
if st.button('Predecir Categoría de Ganancia'):
    # Crear un DataFrame
    df_for_prep = pd.DataFrame({
        'vote_average': [vote_average],
        'vote_count': [vote_count],
        'runtime': [runtime],
        'adult': [adult],
        'budget': [budget],
        'popularity': [popularity],
        'status': [status],
        'original_language': [origen_language]
    })

    # Transformar las características numéricas, booleanas y categóricas
    p_features_array = prep.transform(df_for_prep).toarray()

    # Obtener los nombres de las columnas transformadas por el ColumnTransformer
    allN_from_prep = []
    allN_from_prep.extend(prep.named_transformers_['num_scaled_part1'].get_feature_names_out(['vote_average', 'vote_count', 'runtime']))
    allN_from_prep.extend(prep.named_transformers_['bool_passthrough'].get_feature_names_out(['adult']))
    allN_from_prep.extend(prep.named_transformers_['num_scaled_part2'].get_feature_names_out(['budget', 'popularity']))
    allN_from_prep.extend(prep.named_transformers_['cat_encoded'].get_feature_names_out(XO_names_original))

    #Nota importante: Despues de haberlo intentado mucho me di cuenta que el modelo pone las lombres de las columnas con "_"
    #Se debe sustituir los espacios por guiones bajos
    allN_from_prep_processed = [col.replace(' ', '_') for col in allN_from_prep]
    processed_df = pd.DataFrame(p_features_array, columns=allN_from_prep_processed, index=[0])

    # Procesar géneros con MultiLabelBinarizer
    genre_encoded_input = mlb_genres.transform([genres_input])
    # Se sustituyen los espacios con guiones bajos
    genre_columns_processed = [col.replace(' ', '_') for col in mlb_genres.classes_]
    genre_df_input = pd.DataFrame(genre_encoded_input, columns=genre_columns_processed, index=[0])

    # Procesar compañías de producción con MultiLabelBinarizer
    companies_encoded_input = mlb_companies.transform([production_companies_input])
    # Se sustituyen los espacios con guiones bajos
    company_columns_processed = [col.replace(' ', '_') for col in mlb_companies.classes_]
    companies_df_input = pd.DataFrame(companies_encoded_input, columns=company_columns_processed, index=[0])

    # Procesar resumen de la película (overview) con TF-IDF y SVD
    overview_tfidf = vectorizer.transform([overview_text])
    overview_svd_reduced = svd.transform(overview_tfidf)
    overview_df_input = pd.DataFrame(overview_svd_reduced, columns=list(range(overview_svd_reduced.shape[1])), index=[0])

    # Concatenar todos los DataFrames de características
    final_input_df = pd.concat([processed_df, genre_df_input, companies_df_input, overview_df_input], axis=1)
    final_input_df.columns = final_input_df.columns.astype(str)

    # Poner en orden las columnas
    final_input_df = final_input_df.reindex(columns=model_feature_columns, fill_value=0)

    # Realizar la predicción
    prediction_encoded = automl_model.predict(final_input_df)
    prediction_label = LE.inverse_transform(prediction_encoded)

    st.success(f'La categoría de ganancia predicha es: **{prediction_label[0]}**')
