# TaskAI — Gestor de Tareas con Inteligencia Artificial

> Sistema web de gestión de tareas que utiliza Machine Learning para predecir automáticamente la prioridad de cada tarea, ayudando al usuario a enfocarse en lo que realmente importa.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![XGBoost](https://img.shields.io/badge/XGBoost-✓-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Tabla de contenidos

1. [Problema que resuelve](#-problema-que-resuelve)
2. [Librerías y frameworks utilizados](#-librerías-y-frameworks-utilizados)
3. [Construcción del dataset](#-construcción-del-dataset)
4. [Modelos de Machine Learning](#-modelos-de-machine-learning)
5. [Métricas de efectividad](#-métricas-de-efectividad)
6. [Predicciones del sistema](#-predicciones-del-sistema)
7. [Integración de predicciones en la solución](#-integración-de-predicciones-en-la-solución)
8. [Arquitectura del proyecto](#-arquitectura-del-proyecto)
9. [Explicación del backend](#-explicación-del-backend)
10. [Explicación del frontend](#-explicación-del-frontend)
11. [Cómo las predicciones generan nuevas reglas](#-cómo-las-predicciones-generan-nuevas-reglas)
12. [Cómo correr el proyecto](#-cómo-correr-el-proyecto)
13. [Estructura de archivos](#-estructura-de-archivos)

---

## Problema que resuelve

En el día a día, las personas acumulan múltiples tareas de distinta naturaleza: laborales, personales y académicas. El principal problema no es tener muchas tareas, sino **no saber cuál atender primero**.

Priorizar manualmente implica un juicio subjetivo que muchas veces falla: se tiende a hacer lo fácil antes que lo urgente, o lo visible antes que lo importante. Esto genera:

- Entregas tardías en tareas de alto impacto
- Estrés por mala gestión del tiempo
- Olvido de tareas con fecha límite cercana

**TaskAI** soluciona esto mediante un modelo de Machine Learning entrenado para evaluar objetivamente cada tarea según sus características (urgencia, impacto, dificultad, tiempo estimado, días hasta el vencimiento y tipo) y predecir automáticamente si debe clasificarse como **alta prioridad** o **normal**, notificando al usuario en tiempo real.

---

## Librerías y frameworks utilizados

### Backend (Python)

| Librería       | Versión | Uso                                                              |
| -------------- | ------- | ---------------------------------------------------------------- |
| `flask`        | 3.x     | API REST que expone los endpoints de predicción                  |
| `flask-cors`   | 4.x     | Permite peticiones cross-origin desde React                      |
| `scikit-learn` | 1.4+    | Pipeline, preprocesamiento y métricas de evaluación              |
| `xgboost`      | 2.x     | Modelo principal en producción                                   |
| `lightgbm`     | 4.x     | Modelo candidato en la comparativa                               |
| `pandas`       | 2.x     | Manipulación del dataset y construcción del DataFrame de entrada |
| `numpy`        | 1.26+   | Generación del dataset sintético y operaciones numéricas         |
| `pickle`       | stdlib  | Serialización del pipeline entrenado                             |

### Frontend (JavaScript)

| Librería    | Versión | Uso                          |
| ----------- | ------- | ---------------------------- |
| `react`     | 18      | Interfaz de usuario dinámica |
| `react-dom` | 18      | Renderizado en el navegador  |

> El frontend fue construido con estilos CSS inline para máxima portabilidad, sin dependencia de frameworks de CSS externos.

---

## Construcción del dataset

El dataset fue generado sintéticamente mediante el script `generador_dataset.py`, simulando condiciones realistas de un entorno de gestión de tareas.

### Características generadas

| Feature        | Tipo       | Rango   | Descripción                            |
| -------------- | ---------- | ------- | -------------------------------------- |
| `urgencia`     | Entero     | 1 – 10  | Qué tan urgente es atender la tarea    |
| `dificultad`   | Entero     | 1 – 10  | Complejidad técnica o cognitiva        |
| `tiempo`       | Entero     | 1 – 10  | Horas estimadas para completarla       |
| `fecha_limite` | Entero     | 1 – 10  | Días restantes hasta el vencimiento    |
| `impacto`      | Entero     | 1 – 10  | Consecuencias de no hacerla a tiempo   |
| `tipo`         | Categórico | 0, 1, 2 | Trabajo (0), Personal (1), Estudio (2) |

### Variable objetivo

`prioritaria`: valor binario (0 = normal, 1 = alta prioridad), calculado mediante un score ponderado con ruido gaussiano:

```python
score = (
    urgencia     * 0.30 +
    impacto      * 0.25 +
    (10-tiempo)  * 0.15 +
    (10-fecha_limite) * 0.20 +
    dificultad   * 0.10
)
score_final = score + ruido_gaussiano(media=0, std=2)
prioritaria = 1 si score_final > 6.5 else 0
```

Los pesos reflejan que la urgencia y el impacto son los factores más determinantes, seguidos de la proximidad del vencimiento.

### Tamaño del dataset

- **Total de registros:** 500
- **Tareas prioritarias (1):** ~47% del total
- **Tareas normales (0):** ~53% del total
- **Split de entrenamiento/prueba:** 80% / 20% con estratificación de clases

---

## Modelos de Machine Learning

Se evaluaron 4 modelos de clasificación binaria. Todos fueron envueltos en un **Pipeline de scikit-learn** que incluye `StandardScaler` antes del clasificador, garantizando que las features estén correctamente normalizadas.

### 1. XGBoost (seleccionado para producción)

```
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
```

**Por qué se eligió:** Es el modelo con mayor accuracy y ROC-AUC en validación cruzada. Maneja bien las relaciones no lineales entre features (por ejemplo, una tarea puede ser urgente pero de bajo impacto, y el modelo aprende a ponderar eso correctamente). Es robusto frente al ruido que se introdujo intencionalmente en el dataset.

### 2. LightGBM

Alternativa a XGBoost, más rápida en datasets grandes. Resultados ligeramente inferiores en este caso por el tamaño reducido del dataset (500 registros).

### 3. Random Forest (baseline)

Modelo con el que inició el proyecto. Sirve como baseline sólido e interpretable, pero su accuracy es ~5% menor que XGBoost al no aplicar boosting.

### 4. SVM (Support Vector Machine)

Kernel RBF. Menor rendimiento en este problema por la naturaleza tabular de los datos y las relaciones no lineales entre features numéricas escaladas.

---

## Métricas de efectividad

Evaluación con validación cruzada de 5 folds (`StratifiedKFold`) para garantizar representación balanceada de clases en cada fold.

| Modelo              | CV Accuracy      | Test Accuracy | ROC-AUC   |
| ------------------- | ---------------- | ------------- | --------- |
| **XGBoost** ← mejor | **92.3% ± 1.8%** | **92.0%**     | **0.971** |
| LightGBM            | 91.0% ± 2.1%     | 90.5%         | 0.968     |
| Random Forest       | 87.1% ± 2.4%     | 86.5%         | 0.940     |
| SVM                 | 79.4% ± 3.2%     | 78.0%         | 0.882     |

### Reporte detallado — XGBoost

```
              precision    recall  f1-score   support

   Normal (0)    0.94      0.91      0.92        53
Prioritaria (1)  0.90      0.93      0.92        47

      accuracy                       0.92       100
     macro avg    0.92      0.92     0.92       100
  weighted avg    0.92      0.92     0.92       100
```

### Interpretación

- **Precision 0.90** en tareas prioritarias: el 90% de las tareas marcadas como urgentes realmente lo son.
- **Recall 0.93** en tareas prioritarias: el modelo detecta el 93% de todas las tareas urgentes reales.
- **ROC-AUC 0.971**: el modelo tiene excelente capacidad de discriminación entre clases.

---

## Predicciones del sistema

Ante cada nueva tarea ingresada por el usuario, el sistema realiza una petición `POST /predecir` con los 6 atributos de la tarea. El modelo retorna:

```json
{
  "prioritaria": 1,
  "confianza": 0.9421,
  "probabilidades": {
    "normal": 0.0579,
    "prioritaria": 0.9421
  }
}
```

- `prioritaria`: clasificación binaria (0 o 1)
- `confianza`: probabilidad de la clase predicha (qué tan seguro está el modelo)
- `probabilidades`: distribución completa de probabilidad entre ambas clases

---

## Integración de predicciones en la solución

Las predicciones no solo clasifican la tarea; **generan comportamientos concretos** en la interfaz:

| Predicción           | Comportamiento en el sistema                                            |
| -------------------- | ----------------------------------------------------------------------- |
| `prioritaria: 1`     | Tarea aparece con borde rojo, badge ` Urgente`                          |
| `prioritaria: 0`     | Tarea aparece con borde verde, badge ` Normal`                          |
| Cualquier predicción | Notificación flotante con título, descripción y porcentaje de confianza |
| `prioritaria: 1`     | Se agrega ítem nuevo al panel de notificaciones del sidebar             |
| `prioritaria: 1`     | Aparece en el panel de `Recomendaciones IA`                             |
| Acumulado de tareas  | Las estadísticas del dashboard se recalculan en tiempo real             |
| Acumulado de tareas  | Las barras de distribución de prioridad se actualizan                   |

---

## Arquitectura del proyecto

```
Usuario (navegador)
       │
       │  HTTP (localhost:3000)
       ▼
┌─────────────────────┐
│   Frontend React    │  ← TaskManagerAI.jsx
│   puerto 3000       │
└────────┬────────────┘
         │
         │  POST /predecir
         │  GET  /modelos
         │  GET  /tareas/stats
         │  (HTTP JSON, puerto 5000)
         ▼
┌─────────────────────┐
│   Backend Flask     │  ← app.py
│   puerto 5000       │
└────────┬────────────┘
         │
         │  pipeline.predict()
         ▼
┌─────────────────────┐
│  modelo_entrenado   │  ← StandardScaler + XGBoost
│  .pkl               │     entrenado en dataset_tareas.csv
└─────────────────────┘
```

---

## Explicación del backend

El backend es una **API REST construida con Flask** que expone 3 endpoints:

### `GET /`

Verifica que la API esté activa y el modelo cargado.

### `POST /predecir`

Recibe los atributos de una tarea en JSON, los valida (campos requeridos, tipos numéricos, rangos 1-10), construye un DataFrame con el mismo orden de features del entrenamiento, y retorna la predicción con su nivel de confianza.

**Validaciones implementadas:**

- Presencia de los 6 campos requeridos → `400` si falta alguno
- Tipo numérico de cada campo → `400` si no es número
- Rango válido por campo → `400` con mensaje descriptivo
- Modelo no disponible → `503`
- Error interno → `500`

### `GET /modelos`

Entrena los 4 modelos candidatos en tiempo real con validación cruzada y retorna las métricas comparativas para mostrar en el dashboard.

### `GET /tareas/stats`

Retorna estadísticas descriptivas del dataset (totales, promedios por feature, distribución por tipo).

**Decisiones técnicas clave:**

- Se usa `Pipeline(StandardScaler + XGBClassifier)` para garantizar que el escalado siempre se aplique antes de predecir, evitando data leakage.
- `flask-cors` habilitado para permitir peticiones desde React en `localhost:3000`.
- Todos los errores retornan códigos HTTP semánticamente correctos.

---

## Explicación del frontend

El frontend es una **Single Page Application construida con React 18** que consume la API Flask.

### Componentes principales

| Componente          | Descripción                                                   |
| ------------------- | ------------------------------------------------------------- |
| `TaskManagerAI`     | Componente raíz, maneja el estado global de tareas            |
| `SliderField`       | Input de rango reutilizable con etiqueta de valor             |
| `TaskCard`          | Tarjeta de tarea con prioridad, metadata y barra de confianza |
| `ModelComparison`   | Panel de comparativa de modelos con barras animadas           |
| `ToastNotification` | Notificación flotante que aparece tras cada predicción        |

### Funcionalidades de la interfaz

- **Formulario de nueva tarea** con 6 sliders (1-10) y selector de tipo
- **Predicción en tiempo real** al enviar el formulario, con indicador de carga
- **Notificaciones flotantes** automáticas según el resultado de la IA
- **Panel de notificaciones** en el sidebar con historial de alertas
- **Filtros de lista** por prioridad (todas / alta / normal)
- **Estadísticas en el dashboard** actualizadas en tiempo real
- **Comparativa de modelos** con barras de accuracy animadas
- **Indicador de estado de la API** con fallback a predicción local
- **Eliminación de tareas** con actualización instantánea de métricas

### Manejo de conexión

El frontend detecta si la API está disponible al montar el componente. Si la API no responde, activa automáticamente un modo de predicción local basado en la misma fórmula de score del dataset, permitiendo demostrar la interfaz sin necesidad del backend.

---

## Cómo las predicciones generan nuevas reglas

Las predicciones del modelo no son un dato aislado; se convierten en **reglas de comportamiento** dentro del sistema:

**Regla 1 — Notificación inmediata:**
Si `prioritaria == 1` → mostrar toast con ícono de advertencia y agregar ítem al panel de notificaciones del sidebar.

**Regla 2 — Clasificación visual:**
El color del borde izquierdo de cada tarjeta (rojo/verde) se determina exclusivamente por la predicción del modelo, no por ningún criterio manual del usuario.

**Regla 3 — Panel de recomendaciones:**
El panel lateral "Recomendaciones IA" muestra automáticamente las tareas urgentes ordenadas por fecha límite, generado dinámicamente desde las predicciones acumuladas.

**Regla 4 — Distribución de prioridad:**
Las barras de porcentaje del dashboard reflejan la proporción real de predicciones `0` y `1` en la lista activa, actualizándose con cada nueva tarea.

**Regla 5 — Badge del sidebar:**
El contador de "Alertas" en el menú lateral muestra el número de tareas clasificadas como prioritarias, funcionando como indicador de carga de trabajo urgente.

---

## Cómo correr el proyecto

### Requisitos

- Python 3.10+
- Node.js 18+

### Backend

```bash
# 1. Instalar dependencias
pip install flask flask-cors xgboost lightgbm scikit-learn pandas numpy

# 2. Generar el dataset
python generador_dataset.py

# 3. Entrenar el modelo
python modelo.py

# 4. (Opcional) Ver comparativa de modelos
python analisis.py

# 5. Levantar la API
python app.py
# → http://localhost:5000
```

### Frontend

```bash
# 1. Crear proyecto React
npx create-react-app gestor-tareas-frontend
cd gestor-tareas-frontend

# 2. Copiar TaskManagerAI.jsx a src/
# 3. Reemplazar src/App.js con:
#    import TaskManagerAI from './TaskManagerAI';
#    export default function App() { return <TaskManagerAI />; }

# 4. Correr React
npm start
# → http://localhost:3000
```

### Dos terminales simultáneas

```
Terminal 1 → python app.py       (backend)
Terminal 2 → npm start           (frontend)
```

---

## Estructura de archivos

```
gestor-tareas-ia/
│
├── backend/
│   ├── app.py                  # API REST Flask con endpoints de predicción
│   ├── modelo.py               # Entrenamiento del pipeline XGBoost
│   ├── analisis.py             # Comparativa de modelos y métricas
│   ├── generador_dataset.py    # Generación del dataset sintético
│   ├── dataset_tareas.csv      # Dataset generado (500 registros)
│   └── modelo_entrenado.pkl    # Pipeline serializado (scaler + XGBoost)
│
├── frontend/
│   └── src/
│       ├── App.js              # Punto de entrada React
│       └── TaskManagerAI.jsx   # Componente principal de la aplicación
│
└── README.md
```

---

## Autor

Proyecto desarrollado como entrega final del curso de Inteligencia Artificial Aplicada.

---

_TaskAI — Porque no todas las tareas son iguales._
