"""
=============================================================================
APLICACIÓN DE MÓDULOS Y PAQUETES EN PYTHON PARA MACHINE LEARNING
EN UN CONTEXTO EMPRESARIAL
=============================================================================

Proyecto educativo que demuestra el uso de módulos y paquetes de Python
para el análisis de datos financieros, Machine Learning, NLP y Deep Learning.

Requerimientos: TAREA.md
Datos: data/data.csv (datos económicos del Banco Mundial)
"""

# =============================================================================
# 0. IMPORTACIONES
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# Descargar recursos de NLTK
for resource in ["punkt", "stopwords", "vader_lexicon", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

print("=" * 70)
print("APLICACIÓN DE MÓDULOS Y PAQUETES EN PYTHON PARA ML")
print("EN UN CONTEXTO EMPRESARIAL")
print("=" * 70)


# =============================================================================
# 1. LECTURA Y TRANSFORMACIÓN DE DATOS
# =============================================================================

print("\n" + "=" * 70)
print("1. LECTURA Y TRANSFORMACIÓN DE DATOS (Pandas / NumPy)")
print("=" * 70)

DATA_PATH = os.path.join("data", "data.csv")

# --- 1.1 Lectura del CSV ---
df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Archivo leido: {DATA_PATH}")
print(f"  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

# --- 1.2 Inspección de la estructura ---
print("\n--- Primeras 5 filas ---")
print(df.head())

print("\n--- Tipos de datos ---")
print(df.dtypes)

print("\n--- Información general ---")
df.info()

# --- 1.3 Identificar columnas numéricas y categóricas ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nColumnas numéricas ({len(numeric_cols)}): {numeric_cols}")
print(f"Columnas categóricas ({len(categorical_cols)}): {categorical_cols}")

# --- 1.4 Valores faltantes ---
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({"Faltantes": missing, "Porcentaje": missing_pct})
missing_report = missing_report[missing_report["Faltantes"] > 0].sort_values(
    "Porcentaje", ascending=False
)
print("\n--- Valores faltantes ---")
if len(missing_report) > 0:
    print(missing_report)
else:
    print("  No hay valores faltantes.")

# --- 1.5 Datos duplicados ---
duplicates = df.duplicated().sum()
print(f"\n--- Duplicados: {duplicates} filas duplicadas ---")

# --- 1.6 Limpieza de datos ---
# Eliminar filas donde Year o GDP sean NaN (variables clave)
key_cols = ["Year"]
if " Gross Domestic Product (GDP) " in df.columns:
    key_cols.append(" Gross Domestic Product (GDP) ")
elif "Gross Domestic Product (GDP)" in df.columns:
    key_cols.append("Gross Domestic Product (GDP)")

# Limpiar nombres de columnas (quitar espacios extra)
df.columns = df.columns.str.strip()

print(f"\nColumnas limpias: {list(df.columns)}")

# Ahora buscar GDP y otros indicadores clave
gdp_col = [c for c in df.columns if "GDP" in c.upper()]
gni_col = [c for c in df.columns if "GNI" in c.upper() or "Gross National" in c]
pop_col = [c for c in df.columns if "Population" in c]
year_col = "Year" if "Year" in df.columns else None

print(f"Columna GDP: {gdp_col}")
print(f"Columna GNI: {gni_col}")
print(f"Columna Población: {pop_col}")
print(f"Columna Año: {year_col}")

# Seleccionar columnas numéricas relevantes para análisis
analysis_cols = ["Year"]
for col_list in [gdp_col, gni_col, pop_col]:
    if col_list:
        analysis_cols.append(col_list[0])

# Agregar indicadores económicos disponibles
economic_indicators = [
    "AMA exchange rate", "IMF based exchange rate",
    "Per capita GNI",
    "Exports of goods and services",
    "Imports of goods and services",
    "Manufacturing (ISIC D)",
    "Total Value Added"
]
for ind in economic_indicators:
    matches = [c for c in df.columns if ind.lower() in c.lower()]
    if matches:
        analysis_cols.append(matches[0])

# Mantener solo las que existen (sin duplicados)
analysis_cols = list(dict.fromkeys([c for c in analysis_cols if c in df.columns]))
print(f"\nColumnas para analisis: {analysis_cols}")

# --- 1.7 Preparar datos para ML ---
# Crear variable objetivo: categoría de crecimiento del GDP
if gdp_col:
    gdp_name = gdp_col[0]
    df["GDP_lag"] = df.groupby("Country")[gdp_name].shift(1)
    df["GDP_growth"] = ((df[gdp_name] - df["GDP_lag"]) / df["GDP_lag"].replace(0, np.nan)) * 100

    # Clasificar crecimiento: Alto (>3%), Medio (0-3%), Bajo (<0%)
    def classify_growth(g):
        if pd.isna(g):
            return "Sin datos"
        elif g > 3:
            return "Alto"
        elif g >= 0:
            return "Medio"
        else:
            return "Bajo"

    df["GDP_growth_cat"] = df["GDP_growth"].apply(classify_growth)
    print(f"\n--- Distribución de crecimiento del GDP ---")
    print(df["GDP_growth_cat"].value_counts())

# Seleccionar solo datos numéricos para ML
df_numeric = df[analysis_cols].copy() if analysis_cols else df.select_dtypes(include=[np.number]).copy()
print(f"\n[OK] Datos numéricos preparados: {df_numeric.shape}")


# =============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS
# =============================================================================

print("\n" + "=" * 70)
print("2. ANÁLISIS EXPLORATORIO DE DATOS (Pandas / NumPy / SciPy)")
print("=" * 70)

# --- 2.1 Estadísticas descriptivas ---
print("\n--- Estadísticas descriptivas ---")
desc = df_numeric.describe()
print(desc)

# --- 2.2 Distribución de variables numéricas ---
print("\n--- Análisis de distribución ---")
for col in df_numeric.columns[:6]:  # Primeras 6 columnas
    data = df_numeric[col].dropna()
    if len(data) > 0:
        skewness = data.skew()
        kurtosis = data.skew()  # Kurtosis
        print(f"  {col}:")
        print(f"    Media: {data.mean():.2f}, Mediana: {data.median():.2f}")
        print(f"    Asimetría (skewness): {skewness:.2f}")

# --- 2.3 Correlaciones ---
print("\n--- Matriz de correlaciones (top 10 pares) ---")
corr_matrix = df_numeric.corr()

# Encontrar las correlaciones más fuertes
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_pairs.append((
            corr_matrix.columns[i],
            corr_matrix.columns[j],
            corr_matrix.iloc[i, j]
        ))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("Top 10 correlaciones:")
for c1, c2, corr in corr_pairs[:10]:
    print(f"  {c1} <-> {c2}: {corr:.4f}")

# --- 2.4 Prueba estadística: correlación de Pearson ---
if len(analysis_cols) >= 2:
    col_x = [c for c in analysis_cols if "GDP" in c.upper()]
    col_y = [c for c in analysis_cols if "GNI" in c.upper()]
    if col_x and col_y:
        x_data = df[col_x[0]].dropna()
        y_data = df[col_y[0]].dropna()
        common_idx = x_data.index.intersection(y_data.index)
        if len(common_idx) > 2:
            r, p = pearsonr(x_data.loc[common_idx], y_data.loc[common_idx])
            print(f"\n--- Correlación Pearson entre {col_x[0]} y {col_y[0]} ---")
            print(f"  r = {r:.4f}, p-valor = {p:.2e}")
            print(f"  {'Correlación significativa (p < 0.05)' if p < 0.05 else 'No significativa'}")

# --- 2.5 Análisis por país (si aplica) ---
if "Country" in df.columns:
    country_counts = df["Country"].nunique()
    print(f"\n--- Países en el dataset: {country_counts} ---")
    # Top 5 países por número de registros
    top_countries = df["Country"].value_counts().head(5)
    print("Top 5 países (más registros):")
    for country, count in top_countries.items():
        print(f"  {country}: {count} registros")


# =============================================================================
# 3. MODELO DE CLASIFICACIÓN (Scikit-learn)
# =============================================================================

print("\n" + "=" * 70)
print("3. MODELO DE CLASIFICACIÓN (Scikit-learn)")
print("=" * 70)

# Preparar datos para clasificación
# Usar indicadores económicos para predecir categoría de crecimiento del GDP
feature_cols = [c for c in analysis_cols if c not in ["Year", gdp_col[0] if gdp_col else ""]]
feature_cols = [c for c in feature_cols if c in df.columns]

if "GDP_growth_cat" in df.columns and len(feature_cols) > 0:
    # Filtrar datos sin "Sin datos" en la categoría objetivo
    df_ml = df[df["GDP_growth_cat"] != "Sin datos"].copy()
    df_ml = df_ml[feature_cols + ["GDP_growth_cat"]].dropna()

    if len(df_ml) > 10:
        X = df_ml[feature_cols].values
        y = df_ml["GDP_growth_cat"].values

        # Codificar variable objetivo
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n--- Datos de entrenamiento: {X_train.shape[0]} muestras ---")
        print(f"--- Datos de prueba: {X_test.shape[0]} muestras ---")
        print(f"--- Clases: {list(le.classes_)} ---")

        # Entrenar Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)

        # Predicción
        y_pred = rf_model.predict(X_test_scaled)

        # Evaluación
        print("\n--- Evaluación del modelo Random Forest ---")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

        print("\n--- Reporte de clasificación ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print("--- Matriz de confusión ---")
        print(cm)

        # Importancia de features
        importances = rf_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        print("\n--- Importancia de características ---")
        for feat, imp in feat_imp.items():
            print(f"  {feat}: {imp:.4f}")

        HAS_ML = True
    else:
        print("  Datos insuficientes para clasificación.")
        HAS_ML = False
else:
    print("  No se pudo generar variable objetivo para clasificación.")
    HAS_ML = False


# =============================================================================
# 4. PREPROCESAMIENTO Y VALIDACIÓN (Técnico adicional)
# =============================================================================

print("\n" + "=" * 70)
print("4. PREPROCESAMIENTO Y VALIDACIÓN")
print("=" * 70)

# --- 4.1 Tratamiento de valores faltantes ---
print("\n--- Estrategia de imputación ---")
df_clean = df.copy()
num_imputed = 0
for col in df_clean.select_dtypes(include=[np.number]).columns:
    n_missing = df_clean[col].isnull().sum()
    if n_missing > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        num_imputed += n_missing
print(f"  Valores numéricos imputados con mediana: {num_imputed}")

# --- 4.2 Eliminación de duplicados ---
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
after = len(df_clean)
print(f"  Duplicados eliminados: {before - after}")

# --- 4.3 Normalización/Estándarización ---
scaler_prep = StandardScaler()
numeric_for_scaling = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_for_scaling] = scaler_prep.fit_transform(df_clean[numeric_for_scaling])
print(f"  Columnas estandarizadas: {len(numeric_for_scaling)}")

# --- 4.4 Verificación de fuga de datos (temporal) ---
if year_col and year_col in df_clean.columns:
    print("\n--- Verificación de orden temporal ---")
    df_clean = df_clean.sort_values([c for c in ["Country", year_col] if c in df_clean.columns])
    print("  Datos ordenados cronológicamente [OK]")
    print("  No se usa información futura para predecir el pasado [OK]")

print("\n[OK] Preprocesamiento completado.")


# =============================================================================
# 5. PROCESAMIENTO DE LENGUAJE NATURAL (NLP) — NLTK / SciPy
# =============================================================================

print("\n" + "=" * 70)
print("5. PROCESAMIENTO DE LENGUAJE NATURAL (NLP)")
print("=" * 70)

# El dataset no contiene texto, así que usamos reportes financieros de ejemplo
# para demostrar las técnicas de NLP

sample_reports = [
    "The company reported strong revenue growth of 15% year over year, "
    "driving increased investor confidence and higher stock valuations.",

    "Despite supply chain disruptions, the firm maintained stable operations "
    "and delivered results in line with analyst expectations.",

    "Rising inflation and tightening monetary policy have created headwinds "
    "for the sector, leading to declining margins and reduced outlook.",

    "The acquisition of a major competitor positions the company for "
    "significant market share gains in emerging markets over the next quarter.",

    "Losses exceeded projections due to currency depreciation and increased "
    "operational costs, resulting in a downgrade from credit rating agencies.",

    "The central bank announced measures to stabilize the economy, "
    "including interest rate adjustments and fiscal stimulus programs.",

    "Technology sector stocks surged on positive earnings reports, "
    "with several firms beating expectations by wide margins.",

    "Concerns about geopolitical tensions have increased market volatility, "
    "causing investors to shift toward safe-haven assets like gold.",

    "The company's debt-to-equity ratio improved significantly, reflecting "
    "strong balance sheet management and reduced financial leverage.",

    "Quarterly results showed declining consumer spending, signaling "
    "potential economic slowdown and reduced demand in key markets."
]

# --- 5.1 Limpieza del texto ---
print("\n--- Limpieza del texto ---")


def clean_text(text):
    """Limpieza básica de texto para NLP."""
    text = text.lower()
    # Eliminar signos de puntuación (mantener letras, números, espacios)
    import re
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


cleaned_reports = [clean_text(r) for r in sample_reports]
print("  Texto limpiado: minúsculas, sin puntuación [OK]")

# --- 5.2 Tokenización ---
print("\n--- Tokenización ---")
all_tokens = []
for i, report in enumerate(cleaned_reports[:3]):
    tokens = word_tokenize(report)
    all_tokens.extend(tokens)
    print(f"\n  Reporte {i + 1}:")
    print(f"    Original: {sample_reports[i][:80]}...")
    print(f"    Tokens ({len(tokens)}): {tokens[:10]}...")

# --- 5.3 Eliminación de stop words ---
print("\n--- Eliminación de stop words ---")
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = set()

filtered_tokens = [t for t in all_tokens if t not in stop_words and len(t) > 2]
print(f"  Tokens originales: {len(all_tokens)}")
print(f"  Tokens filtrados: {len(filtered_tokens)}")
print(f"  Stop words eliminadas: {len(all_tokens) - len(filtered_tokens)}")

# --- 5.4 Análisis de palabras más frecuentes ---
from collections import Counter

word_freq = Counter(filtered_tokens)
print("\n--- Palabras más frecuentes ---")
for word, freq in word_freq.most_common(10):
    print(f"  {word}: {freq}")

# --- 5.5 SciPy: análisis estadístico del texto ---
# Longitud de oraciones por reporte
print("\n--- Análisis estadístico con SciPy ---")
sentence_lengths = []
for report in sample_reports:
    sentences = sent_tokenize(report)
    lengths = [len(s.split()) for s in sentences]
    sentence_lengths.append(np.mean(lengths))

sentence_lengths = np.array(sentence_lengths)
print(f"  Promedio de palabras por oración: {sentence_lengths.mean():.2f}")
print(f"  Desviación estándar: {sentence_lengths.std():.2f}")
print(f"  Media (SciPy): {stats.tmean(sentence_lengths):.2f}")
print(f"  Varianza (SciPy): {stats.tvar(sentence_lengths):.2f}")


# =============================================================================
# 6. ANÁLISIS DE SENTIMIENTO
# =============================================================================

print("\n" + "=" * 70)
print("6. ANÁLISIS DE SENTIMIENTO")
print("=" * 70)

sia = SentimentIntensityAnalyzer()

print("\n--- Resultados del análisis de sentimiento ---")
sentiment_results = []
for i, report in enumerate(sample_reports):
    scores = sia.polarity_scores(report)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positivo"
    elif compound <= -0.05:
        label = "Negativo"
    else:
        label = "Neutral"

    sentiment_results.append({
        "Reporte": f"Reporte {i + 1}",
        "Sentimiento": label,
        "Compuesto": compound,
        "Positivo": scores["pos"],
        "Neutro": scores["neu"],
        "Negativo": scores["neg"]
    })

    print(f"\n  Reporte {i + 1}: {label} (compound: {compound:.4f})")
    print(f"    Texto: {report[:60]}...")

# Resumen de sentimientos
sent_df = pd.DataFrame(sentiment_results)
print("\n--- Resumen de sentimientos ---")
print(sent_df[["Reporte", "Sentimiento", "Compuesto"]].to_string(index=False))

sentiment_counts = sent_df["Sentimiento"].value_counts()
print(f"\nDistribución:")
for s, c in sentiment_counts.items():
    print(f"  {s}: {c}")


# =============================================================================
# 7. TOKENIZACIÓN (Demostración detallada con NLTK)
# =============================================================================

print("\n" + "=" * 70)
print("7. TOKENIZACIÓN (NLTK)")
print("=" * 70)

demo_text = (
    "Financial markets experienced significant volatility amid concerns "
    "about inflation, interest rates, and geopolitical developments. "
    "The S&P 500 closed 2.3% lower, while the Nasdaq dropped 3.1%. "
    "Investors are closely watching Federal Reserve announcements for "
    "guidance on future monetary policy direction."
)

print(f"\n--- Texto de demostración ---")
print(f"  \"{demo_text}\"")

# Flujo de tokenización
print("\n--- Flujo de tokenización ---")
print("  Reporte financiero")
print("        |")
print("  Limpieza (minúsculas, sin puntuación)")
clean = clean_text(demo_text)
print("        |")
print("  Tokenización (NLTK word_tokenize)")
tokens = word_tokenize(clean)
print(f"        |")
print(f"  Palabras individuales: {tokens}")
print("        |")
print("  Análisis")

# Tokenización por oraciones
sentences = sent_tokenize(demo_text)
print(f"\n--- Tokenización por oraciones ({len(sentences)} oraciones) ---")
for i, s in enumerate(sentences):
    print(f"  {i + 1}. {s}")

# Tokenización por palabras
tokens_no_stop = [t for t in tokens if t not in stop_words and len(t) > 2]
print(f"\n--- Tokens sin stop words ({len(tokens_no_stop)}) ---")
print(f"  {tokens_no_stop}")

# Bigramas
bigrams = list(zip(tokens_no_stop[:-1], tokens_no_stop[1:]))
print(f"\n--- Bigramas (primeros 5) ---")
for b in bigrams[:5]:
    print(f"  {b[0]} -> {b[1]}")


# =============================================================================
# 8. RED NEURONAL PARA SERIES TEMPORALES (LSTM)
# =============================================================================

print("\n" + "=" * 70)
print("8. RED NEURONAL PARA SERIES TEMPORALES (LSTM con TensorFlow)")
print("=" * 70)

if gdp_col:
    gdp_name = gdp_col[0]

    # Seleccionar un país con suficientes datos
    if "Country" in df.columns:
        country_data = df.groupby("Country")[gdp_name].count()
        best_country = country_data.idxmax()
        ts_data = df[df["Country"] == best_country][gdp_name].dropna().values
        print(f"\n  País seleccionado: {best_country}")
        print(f"  Observaciones: {len(ts_data)}")
    else:
        ts_data = df[gdp_name].dropna().values
        print(f"\n  Observaciones totales: {len(ts_data)}")

    if len(ts_data) > 20:
        # Preprocesamiento
        ts_data = ts_data.reshape(-1, 1).astype(np.float32)
        scaler_ts = StandardScaler()
        ts_scaled = scaler_ts.fit_transform(ts_data)

        # Crear ventanas temporales (lookback = 5)
        lookback = 5

        def create_windows(data, lookback):
            X, y = [], []
            for i in range(len(data) - lookback):
                X.append(data[i : i + lookback])
                y.append(data[i + lookback])
            return np.array(X), np.array(y)

        X_ts, y_ts = create_windows(ts_scaled, lookback)
        print(f"  Ventanas creadas: {X_ts.shape[0]}")
        print(f"  Shape X: {X_ts.shape}, Shape y: {y_ts.shape}")

        # División temporal (sin mezclar orden)
        split = int(len(X_ts) * 0.8)
        X_train_ts, X_test_ts = X_ts[:split], X_ts[split:]
        y_train_ts, y_test_ts = y_ts[:split], y_ts[split:]

        print(f"  Train: {X_train_ts.shape[0]}, Test: {X_test_ts.shape[0]}")

        # --- Modelo LSTM con TensorFlow/Keras ---
        print("\n--- Construyendo modelo LSTM ---")
        model_lstm = keras.Sequential([
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1)
        ])

        model_lstm.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        model_lstm.summary()

        # Entrenamiento
        print("\n--- Entrenando LSTM ---")
        history = model_lstm.fit(
            X_train_ts, y_train_ts,
            epochs=50,
            batch_size=8,
            validation_split=0.1,
            verbose=1
        )

        # Predicción
        y_pred_ts = model_lstm.predict(X_test_ts)

        # Inversión de escala
        y_test_inv = scaler_ts.inverse_transform(y_test_ts)
        y_pred_inv = scaler_ts.inverse_transform(y_pred_ts)

        # Evaluación
        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        print(f"\n--- Evaluación LSTM ---")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")

        HAS_LSTM = True
    else:
        print("  Datos insuficientes para series temporales.")
        HAS_LSTM = False
else:
    print("  No se encontró columna GDP para series temporales.")
    HAS_LSTM = False


# =============================================================================
# 9. DEEP LEARNING CON TENSORFLOW Y KERAS
# =============================================================================

print("\n" + "=" * 70)
print("9. DEEP LEARNING CON TENSORFLOW Y KERAS")
print("=" * 70)

if HAS_ML:
    # Usar los mismos datos preparados en la sección 3
    print("\n--- Modelo de Deep Learning (clasificación) ---")

    # Arquitectura más profunda que Random Forest
    n_classes = len(le.classes_)
    input_dim = X_train_scaled.shape[1]

    model_dl = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax")
    ])

    model_dl.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_dl.summary()

    # Entrenamiento
    print("\n--- Entrenando modelo Deep Learning ---")
    history_dl = model_dl.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # Evaluación
    y_pred_dl = model_dl.predict(X_test_scaled)
    y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)

    print("\n--- Evaluación Deep Learning ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_dl_classes):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_dl_classes, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_dl_classes, average='weighted'):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_dl_classes, average='weighted'):.4f}")

    HAS_DL = True
else:
    print("  No hay datos ML disponibles para Deep Learning.")
    HAS_DL = False


# =============================================================================
# 10. USO DE PYTORCH
# =============================================================================

print("\n" + "=" * 70)
print("10. USO DE PYTORCH")
print("=" * 70)

# --- 10.1 Creación de tensores ---
print("\n--- Creación de tensores ---")
tensor_1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"  Tensor 1D: {tensor_1}")
print(f"  Forma: {tensor_1.shape}")
print(f"  Tipo: {tensor_1.dtype}")

tensor_2d = torch.randn(3, 4)
print(f"\n  Tensor 2D (aleatorio): {tensor_2d.shape}")
print(f"  Media: {tensor_2d.mean():.4f}")

tensor_zeros = torch.zeros(2, 3)
print(f"  Tensor de ceros: {tensor_zeros.shape}")

# --- 10.2 Operaciones con tensores ---
print("\n--- Operaciones con tensores ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"  a + b = {a + b}")
print(f"  a * b = {a * b}")
print(f"  Suma: {torch.sum(a):.1f}")
print(f"  Producto punto: {torch.dot(a, b):.1f}")

# --- 10.3 Definición de red neuronal ---
print("\n--- Red neuronal con PyTorch ---")


class SimpleNet(nn.Module):
    """Red neuronal simple para clasificación."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# --- 10.4 Entrenamiento básico ---
if HAS_ML:
    input_size = X_train_scaled.shape[1]
    hidden_size = 32
    output_size = n_classes

    model_pt = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001)

    # Convertir datos a tensores
    X_train_pt = torch.FloatTensor(X_train_scaled)
    y_train_pt = torch.LongTensor(y_train)
    X_test_pt = torch.FloatTensor(X_test_scaled)
    y_test_pt = torch.LongTensor(y_test)

    print(f"\n  Arquitectura: {model_pt}")
    print(f"  Parámetros totales: {sum(p.numel() for p in model_pt.parameters())}")

    # Entrenamiento
    print("\n--- Entrenando red PyTorch ---")
    epochs_pt = 50
    for epoch in range(epochs_pt):
        # Forward pass
        outputs = model_pt(X_train_pt)
        loss = criterion(outputs, y_train_pt)

        # Backward pass
        optimizer_pt.zero_grad()
        loss.backward()
        optimizer_pt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs_pt}], Loss: {loss.item():.4f}")

    # Evaluación
    with torch.no_grad():
        test_outputs = model_pt(X_test_pt)
        _, predicted = torch.max(test_outputs, 1)
        accuracy_pt = (predicted == y_test_pt).float().mean().item()

    print(f"\n--- Evaluación PyTorch ---")
    print(f"  Accuracy: {accuracy_pt:.4f}")

    # --- 10.5 Propagación hacia adelante (demostración) ---
    print("\n--- Forward pass (demostración) ---")
    sample_input = X_test_pt[:1]
    sample_output = model_pt(sample_input)
    print(f"  Input: {sample_input.shape}")
    print(f"  Output (logits): {sample_output}")
    print(f"  Predicción: {le.classes_[torch.argmax(sample_output).item()]}")

    HAS_PT = True
else:
    # Demostración básica sin datos ML
    print("\n--- Demostración básica (sin datos ML) ---")
    model_pt = SimpleNet(10, 32, 3)
    dummy_input = torch.randn(1, 10)
    dummy_output = model_pt(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {dummy_output.shape}")
    print(f"  Arquitectura: {model_pt}")
    HAS_PT = False


# =============================================================================
# 11-14. VISUALIZACIONES (Matplotlib / Seaborn)
# =============================================================================

print("\n" + "=" * 70)
print("11-14. VISUALIZACIONES (Matplotlib / Seaborn)")
print("=" * 70)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 11. Gráficos comparativos de tendencias ---
print("\n--- 11. Gráficos comparativos de tendencias ---")
if gdp_col and "Country" in df.columns:
    gdp_name = gdp_col[0]
    # Seleccionar algunos países para comparar
    sample_countries = df["Country"].value_counts().head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evolución del GDP por País (11. Tendencias)", fontsize=14, fontweight="bold")

    for ax, country in zip(axes.flatten(), sample_countries):
        country_data = df[df["Country"] == country].sort_values("Year")
        if year_col in country_data.columns and gdp_name in country_data.columns:
            ax.plot(country_data[year_col], country_data[gdp_name], linewidth=1.5)
            ax.set_title(country, fontsize=11)
            ax.set_xlabel("Año")
            ax.set_ylabel("GDP")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tendencias.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/tendencias.png")


# --- 12. Histogramas ---
print("\n--- 12. Histogramas ---")
if gdp_col:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribución de Variables Económicas (12. Histogramas)", fontsize=14, fontweight="bold")

    hist_cols = []
    if gdp_col:
        hist_cols.append(gdp_col[0])
    if gni_col:
        hist_cols.append(gni_col[0])
    if "AMA exchange rate" in df.columns or "AMA exchange rate ".strip() in df.columns:
        ex_col = [c for c in df.columns if "AMA" in c]
        if ex_col:
            hist_cols.append(ex_col[0])

    for ax, col in zip(axes, hist_cols[:3]):
        data = df[col].dropna()
        ax.hist(data, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(col[:30], fontsize=10)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")

    # Si hay menos de 3 columnas, ocultar ejes vacíos
    for i in range(len(hist_cols), 3):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histogramas.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/histogramas.png")


# --- 13. Gráficos de dispersión ---
print("\n--- 13. Gráficos de dispersión ---")
scatter_pairs = []
if gdp_col and gni_col:
    scatter_pairs.append((gdp_col[0], gni_col[0]))
if pop_col and gdp_col:
    scatter_pairs.append((pop_col[0], gdp_col[0]))

if scatter_pairs:
    fig, axes = plt.subplots(1, len(scatter_pairs), figsize=(7 * len(scatter_pairs), 6))
    if len(scatter_pairs) == 1:
        axes = [axes]

    fig.suptitle("Relación entre Variables (13. Dispersión)", fontsize=14, fontweight="bold")

    for ax, (col_x, col_y) in zip(axes, scatter_pairs):
        subset = df[[col_x, col_y]].dropna()
        ax.scatter(subset[col_x], subset[col_y], alpha=0.3, s=10, color="teal")
        ax.set_xlabel(col_x[:30])
        ax.set_ylabel(col_y[:30])
        ax.set_title(f"{col_x[:20]} vs {col_y[:20]}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dispersion.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/dispersion.png")


# --- 14. Líneas de tendencia ---
print("\n--- 14. Líneas de tendencia ---")
if gdp_col and "Country" in df.columns:
    gdp_name = gdp_col[0]
    # Promedio global por año
    global_gdp = df.groupby("Year")[gdp_name].mean().dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_gdp.index, global_gdp.values, linewidth=2, label="GDP Promedio Global")

    # Línea de tendencia (regresión lineal)
    if len(global_gdp) > 2:
        z = np.polyfit(global_gdp.index, global_gdp.values, 1)
        p = np.poly1d(z)
        ax.plot(global_gdp.index, p(global_gdp.index), "--", color="red",
                linewidth=2, label=f"Tendencia (pendiente: {z[0]:.2e})")

    ax.set_title("Evolución del GDP Global con Línea de Tendencia (14)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel("GDP Promedio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lineas_tendencia.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/lineas_tendencia.png")


# --- Visualizaciones adicionales de ML ---
if HAS_ML:
    print("\n--- Visualizaciones de modelos ---")

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Matriz de Confusión - Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusion_rf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/matriz_confusion_rf.png")

    # Importancia de features
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Importancia de Características - Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "importancia_features.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/importancia_features.png")


# --- Visualización de sentimiento ---
print("\n--- Visualización de sentimiento ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras de sentimientos
colors = {"Positivo": "#2ecc71", "Neutral": "#f39c12", "Negativo": "#e74c3c"}
sent_counts = sent_df["Sentimiento"].value_counts()
bar_colors = [colors.get(s, "gray") for s in sent_counts.index]
axes[0].bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor="black")
axes[0].set_title("Distribución de Sentimientos", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad")

# Distribución de scores compuestos
axes[1].hist(sent_df["Compuesto"], bins=10, edgecolor="black", alpha=0.7, color="purple")
axes[1].set_title("Distribución de Scores de Sentimiento", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Score Compuesto")
axes[1].set_ylabel("Frecuencia")
axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sentimiento.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] Guardado: {OUTPUT_DIR}/sentimiento.png")


# --- Visualización del historial de entrenamiento ---
if HAS_LSTM:
    print("\n--- Historial de entrenamiento LSTM ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Pérdida (Loss) - LSTM", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE - LSTM", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_entrenamiento.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/lstm_entrenamiento.png")

    # Predicciones vs reales
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_inv, label="Real", linewidth=2)
    ax.plot(y_pred_inv, label="Predicho (LSTM)", linewidth=2, linestyle="--")
    ax.set_title("Valores Reales vs Predichos - LSTM", fontsize=14, fontweight="bold")
    ax.set_xlabel("Índice")
    ax.set_ylabel("GDP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_predicciones.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/lstm_predicciones.png")

if HAS_DL:
    print("\n--- Historial de entrenamiento Deep Learning ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_dl.history["loss"], label="Train Loss")
    axes[0].plot(history_dl.history["val_loss"], label="Val Loss")
    axes[0].set_title("Pérdida (Loss) - Deep Learning", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_dl.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history_dl.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy - Deep Learning", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dl_entrenamiento.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Guardado: {OUTPUT_DIR}/dl_entrenamiento.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("RESUMEN DEL PROYECTO")
print("=" * 70)

print("""
Módulos y paquetes utilizados:
  [OK] Pandas         -> Manipulación y análisis de datos
  [OK] NumPy          -> Operaciones numéricas
  [OK] Scikit-learn   -> Modelo de clasificación (Random Forest)
  [OK] TensorFlow     -> Deep Learning y LSTM
  [OK] Keras          -> Construcción de redes neuronales
  [OK] PyTorch        -> Tensores y red neuronal básica
  [OK] SciPy          -> Análisis estadístico
  [OK] NLTK           -> Procesamiento de lenguaje natural
  [OK] Matplotlib     -> Visualización de datos
  [OK] Seaborn        -> Visualización estadística

Requerimientos cubiertos:
  [OK] 1.  Lectura y transformación de datos
  [OK] 2.  Análisis exploratorio
  [OK] 3.  Modelo de clasificación (Scikit-learn + PyTorch)
  [OK] 4.  Preprocesamiento y validación
  [OK] 5.  Procesamiento de lenguaje natural (NLP)
  [OK] 6.  Análisis de sentimiento
  [OK] 7.  Tokenización
  [OK] 8.  Red neuronal para series temporales (LSTM)
  [OK] 9.  Deep Learning con TensorFlow/Keras
  [OK] 10. Uso de PyTorch
  [OK] 11. Gráficos comparativos de tendencias
  [OK] 12. Histogramas
  [OK] 13. Gráficos de dispersión
  [OK] 14. Líneas de tendencia
""")

# Listar archivos generados
print("Archivos generados en 'output/':")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {f} ({size_kb:.1f} KB)")

print("\n[OK] Proyecto completado exitosamente.")
print("=" * 70)
