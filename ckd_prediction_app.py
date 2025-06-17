
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

st.title("Prediksi Penyakit Ginjal Kronis (CKD)")

uploaded_file = st.file_uploader("Upload file CSV dataset CKD", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Informasi Dataset")
    st.write(data.info())
    st.dataframe(data.head())

    # Lakukan preprocessing sederhana (contoh)
    data = data.drop(['id'], axis=1, errors='ignore')
    data.replace("?", np.nan, inplace=True)

    # Label encoding & imputasi
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data)

    X = data_imputed[:, :-1]
    y = data_imputed[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with st.spinner("Melatih model..."):
        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[es], verbose=0)

    st.success("Model selesai dilatih.")
    st.write("Akurasi pada data uji:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Akurasi: {acc:.2f}")

    y_pred = model.predict(X_test) > 0.5
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred.astype(int)))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred.astype(int)))
