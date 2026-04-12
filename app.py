# Agregar después de la línea 23:
model_feature_names = joblib.load('model_feature_names.pkl')

# Cambiar las líneas 105-115 por:
processed_df.index = [0]
genre_df_input.index = [0]
companies_df_input.index = [0]
overview_df_input.index = [0]

final_input_df = pd.concat([processed_df, genre_df_input, companies_df_input, overview_df_input], axis=1)
final_input_df.columns = final_input_df.columns.astype(str)

# ⭐ AQUÍ ES LO IMPORTANTE: Asegurar que tenga exactamente las mismas columnas
# Agregar columnas faltantes con 0 si es necesario
for col in model_feature_names:
    if col not in final_input_df.columns:
        final_input_df[col] = 0

# Seleccionar solo las columnas que el modelo espera, en el orden correcto
final_input_df = final_input_df[model_feature_names]

# Realizar la predicción
prediction_encoded = automl_model.predict(final_input_df)
prediction_label = LE.inverse_transform(prediction_encoded)

st.success(f'La categoría de ganancia predicha es: **{prediction_label[0]}**')
