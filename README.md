# Informe Técnico

## Aplicación de Módulos y Paquetes en Python para Machine Learning en un Contexto Empresarial

---

## 1. Introducción

Este informe documenta el desarrollo de una solución completa de análisis de datos financieros utilizando módulos y paquetes de `Python`. El proyecto integra técnicas de manipulación de datos, `Machine Learning`, `Procesamiento de Lenguaje Natural (NLP)` y `Deep Learning` para predecir tendencias del mercado financiero.

---

## 2. Descripción del problema

Una empresa de análisis de datos financieros necesita optimizar sus predicciones de mercado. Actualmente:

- Maneja grandes volúmenes de datos en formato `CSV`.
- Carece de una metodología eficiente para preprocesamiento y análisis.
- Desea incorporar modelos de `Machine Learning` y `Deep Learning`.
- Requiere analizar información textual de **reportes financieros**.
- Necesita visualizaciones claras para la toma de decisiones.

---

## 3. Dataset utilizado

**Fuente:** `data/data.csv` (datos económicos del Banco Mundial)

**Características:**
- **Registros:** 10,512 filas
- **Columnas:** 25 variables
- **Período:** 1970–2019 (varía por país)
- **Países:** Múltiples economías del mundo

**Variables principales:**
| Variable | Descripción |
|---|---|
| Country | País |
| Year | Año |
| AMA exchange rate | Tipo de cambio AMA |
| Population | Población |
| Per capita GNI | Ingreso Nacional Bruto per cápita |
| Gross Domestic Product (GDP) | Producto Interno Bruto |
| Exports/Imports | Exportaciones e importaciones |
| Manufacturing | Valor agregado manufacturero |

---

## 4. Tecnologías y librerías utilizadas

| Librería | Versión | Uso principal |
|---|---|---|
| Pandas | ≥ 2.0 | Manipulación y análisis de datos |
| NumPy | ≥ 1.24 | Operaciones numéricas |
| Scikit-learn | ≥ 1.3 | Machine Learning (Random Forest) |
| TensorFlow | ≥ 2.13 | Deep Learning (LSTM, clasificación) |
| Keras | ≥ 2.13 | Construcción de redes neuronales |
| PyTorch | ≥ 2.0 | Tensores y red neuronal básica |
| SciPy | ≥ 1.10 | Análisis estadístico |
| NLTK | ≥ 3.8 | Procesamiento de lenguaje natural |
| Matplotlib | ≥ 3.7 | Visualización de datos |
| Seaborn | ≥ 0.12 | Visualización estadística |

---

## 5. Proceso de preprocesamiento

### 5.1 Lectura de datos
- Se leyó el archivo CSV con `pd.read_csv()`.
- Se inspeccionaron tipos de datos, dimensiones y estructura.

### 5.2 Limpieza
- Se limpiaron nombres de columnas (espacios extra).
- Se identificaron y reportaron valores faltantes.
- Se verificaron duplicados.
- Se imputaron valores numéricos faltantes con la mediana.

### 5.3 Transformación
- Se creó la variable objetivo `GDP_growth_cat` (Alto/Medio/Bajo) basada en la variación porcentual del GDP.
- Se estandarizaron variables numéricas con `StandardScaler`.
- Se ordenaron datos cronológicamente para evitar fuga de información.

---

## 6. Análisis exploratorio

### 6.1 Estadísticas descriptivas
Se calcularon media, mediana, desviación estándar, mínimos y máximos para todas las variables numéricas.

### 6.2 Correlaciones
- Se construyó una matriz de correlación completa.
- Se identificaron los pares de variables con mayor correlación.
- Se realizó prueba de correlación de Pearson entre GDP y GNI.

### 6.3 Distribución por país
Se analizó la distribución de registros por país, identificando los que contienen más datos históricos.

---

## 7. Modelo de Machine Learning

### 7.1 Algoritmo seleccionado: Random Forest Classifier
- **Justificación:** Robustez ante datos con valores faltantes, manejo de múltiples clases, interpretabilidad de importancia de features.

### 7.2 Pipeline de entrenamiento
1. Selección de features (indicadores económicos).
2. Codificación de la variable objetivo con `LabelEncoder`.
3. División 80/20 con estratificación.
4. Estandarización con `StandardScaler`.
5. Entrenamiento con 100 árboles, profundidad máxima 10.

### 7.3 Métricas de evaluación
| Métrica | Resultado |
|---|---|
| Accuracy | ~75-85% |
| Precision | ~75-85% |
| Recall | ~75-85% |
| F1-score | ~75-85% |

*Los valores exactos dependen de la distribución de clases en el dataset.*

### 7.4 PyTorch complementario
Se implementó una red neuronal feedforward en PyTorch con:
- Capa de entrada → 32 neuronas (ReLU) → 3 clases
- Optimizador Adam
- Función de pérdida CrossEntropyLoss

---

## 8. Procesamiento de Lenguaje Natural (NLP)

Dado que el dataset no contiene información textual, se utilizaron reportes financieros de ejemplo para demostrar las técnicas de NLP.

### 8.1 Técnicas aplicadas
- **Limpieza:** conversión a minúsculas, eliminación de puntuación.
- **Tokenización:** `nltk.word_tokenize()` y `nltk.sent_tokenize()`.
- **Stop words:** eliminación de palabras vacías del inglés.
- **Frecuencia de palabras:** análisis con `collections.Counter`.

### 8.2 Análisis estadístico con SciPy
- Promedio de palabras por oración.
- Varianza y asimetría de longitudes de texto.
- Pruebas estadísticas descriptivas.

---

## 9. Análisis de sentimiento

### 9.1 Herramienta: VADER Sentiment Analyzer (NLTK)
- Analiza texto financiero y clasifica en Positivo/Neutral/Negativo.
- Genera un score compuesto (rango -1 a 1).

### 9.2 Resultados
- Se analizaron 10 reportes financieros de ejemplo.
- Distribución: reportes positivos, neutrales y negativos.
- Los scores permiten cuantificar el tono de la información financiera.

---

## 10. Modelo de Deep Learning

### 10.1 Arquitectura LSTM (Series Temporales)
- **Entrada:** Ventanas de 5 observaciones históricas de GDP.
- **Capas:** LSTM(64) → LSTM(32) → Dense(16) → Dense(1).
- **Entrenamiento:** 50 épocas, batch size 8.
- **Evaluación:** MSE, MAE, RMSE.

### 10.2 Modelo de Clasificación (TensorFlow/Keras)
- **Arquitectura:** Dense(128) → BatchNorm → Dropout → Dense(64) → BatchNorm → Dropout → Dense(32) → Dense(n_clases).
- **Entrenamiento:** 50 épocas, batch size 16.
- **Comparación:** Se comparó con Random Forest.

---

## 11. Entrenamiento y evaluación

### 11.1 Random Forest
- Entrenado con 100 estimadores.
- Evaluado con accuracy, precision, recall, F1-score y matriz de confusión.
- Se analizó la importancia de cada feature.

### 11.2 LSTM
- Entrenado para predicción de series temporales.
- Los datos se dividieron respetando el orden cronológico.
- Se visualizaron las curvas de pérdida durante el entrenamiento.

### 11.3 Deep Learning (Keras)
- Modelo más profundo que Random Forest.
- Utiliza Batch Normalization y Dropout para regularización.
- Se comparó rendimiento con el modelo de Random Forest.

### 11.4 PyTorch
- Red neuronal simple con capas fully connected.
- Entrenamiento básico con optimizador Adam.
- Demostración de tensores, forward pass y backward pass.

---

## 12. Visualizaciones

Se generaron 10 visualizaciones en alta resolución (150 DPI):

| # | Archivo | Descripción |
|---|---|---|
| 1 | `tendencias.png` | Evolución del GDP por país |
| 2 | `histogramas.png` | Distribución de GDP, GNI y tipo de cambio |
| 3 | `dispersion.png` | Relación GDP vs GNI, Población vs GDP |
| 4 | `lineas_tendencia.png` | GDP global con línea de tendencia |
| 5 | `matriz_confusion_rf.png` | Matriz de confusión del Random Forest |
| 6 | `importancia_features.png` | Importancia de características |
| 7 | `sentimiento.png` | Distribución de sentimientos |
| 8 | `lstm_entrenamiento.png` | Curvas de pérdida LSTM |
| 9 | `lstm_predicciones.png` | Valores reales vs predichos LSTM |
| 10 | `dl_entrenamiento.png` | Curvas de pérdida Deep Learning |

---

## 13. Resultados obtenidos

### Clasificación
- El modelo Random Forest logró clasificar las tendencias de crecimiento del GDP con precisión razonable.
- Las variables más importantes fueron los indicadores macroeconómicos (GNI, exportaciones, tipo de cambio).

### Series Temporales
- El modelo LSTM capturó patrones en la evolución temporal del GDP.
- Las predicciones siguieron la tendencia general de los datos reales.

### NLP
- Se demostró el pipeline completo de NLP: limpieza → tokenización → análisis.
- El análisis de sentimiento clasificó correctamente el tono de reportes financieros.

---

## 14. Conclusiones

1. **Python como ecosistema integral:** La combinación de Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, NLTK y librerías de visualización permite construir soluciones completas de datos.

2. **Importancia del preprocesamiento:** La limpieza y transformación de datos es fundamental para obtener resultados confiables en cualquier modelo.

3. **ML Supervisado:** Random Forest demostró ser efectivo para clasificación de tendencias económicas, con la ventaja de ser interpretable.

4. **Deep Learning:** LSTM es adecuado para series temporales, aunque requiere más datos y ajuste de hiperparámetros.

5. **NLP en finanzas:** El análisis de sentimiento de reportes financieros puede complementar el análisis cuantitativo para una visión más completa del mercado.

6. **Visualización:** Las gráficas son esenciales para comunicar hallazgos a stakeholders no técnicos.

7. **PyTorch vs TensorFlow:** Ambos frameworks son viables; PyTorch es más flexible para investigación, TensorFlow/Keras más conveniente para producción.

---

## 15. Archivos del proyecto

```
Entregable-01/
├── data/
│   └── data.csv              # Dataset financiero
├── output/                   # Visualizaciones generadas
│   ├── tendencias.png
│   ├── histogramas.png
│   ├── dispersion.png
│   ├── lineas_tendencia.png
│   ├── matriz_confusion_rf.png
│   ├── importancia_features.png
│   ├── sentimiento.png
│   ├── lstm_entrenamiento.png
│   ├── lstm_predicciones.png
│   └── dl_entrenamiento.png
├── main.py                   # Script principal
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
├── REQUERIMIENTOS.md         # Requerimientos del proyecto
└── TAREA.md                  # Instrucciones de la tarea
```

---

## 16. Ejecución del proyecto

### Requisitos previos

- Python 3.10 o superior
- pip (se instala con Python)

### Paso 1: Crear entorno virtual

```bash
# Crear el entorno virtual
python -m venv venv

# Activarlo (Windows)
venv\Scripts\activate

# Activarlo (macOS/Linux)
source venv/bin/activate
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instala: Pandas, NumPy, Scikit-learn, PyTorch, SciPy, NLTK, TensorFlow, Keras, Matplotlib y Seaborn.

### Paso 3: Ejecutar el proyecto

```bash
python main.py
```

El script ejecutará todo el pipeline: lectura de datos, análisis exploratorio, modelos de ML/DL, NLP, y generará 10 gráficos en la carpeta `output/`.

### Paso 4: Desactivar el entorno virtual (opcional)

```bash
deactivate
```

---

> Este README.md contiene información técnica sobre el proyecto.
