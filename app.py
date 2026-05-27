"""
app.py - API REST para el Gestor de Tareas con IA
Flask + XGBoost 

Endpoints:
  GET  /              → Estado de la API
  POST /predecir      → Predice si una tarea es prioritaria
  GET  /modelos       → Compara múltiples modelos con métricas
  GET  /tareas/stats  → Estadísticas generales del dataset

Instalar dependencias:
  pip install flask flask-cors xgboost scikit-learn pandas lightgbm
"""

import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Inicializar app ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Permite peticiones desde el frontend React (localhost:3000)

# ─── Cargar modelo entrenado (pipeline: scaler + XGBoost) ────────────────────
try:
    with open("modelo_entrenado.pkl", "rb") as f:
        modelo = pickle.load(f)
    print("Modelo cargado correctamente")
except FileNotFoundError:
    modelo = None
    print(" modelo_entrenado.pkl no encontrado. Ejecuta modelo.py primero.")

# ─── Constantes de validación ─────────────────────────────────────────────────
CAMPOS_REQUERIDOS = ["urgencia", "dificultad", "tiempo", "fecha_limite", "impacto", "tipo"]

RANGOS = {
    "urgencia":     (1, 10),
    "dificultad":   (1, 10),
    "tiempo":       (1, 10),
    "fecha_limite": (1, 10),
    "impacto":      (1, 10),
    "tipo":         (0, 2),
}


# ─── Utilidad: validar campos del request ─────────────────────────────────────
def validar_campos(data: dict):
    """
    Verifica que todos los campos requeridos existan y estén en rango.
    Retorna (None, None) si todo está bien, o (mensaje_error, código_http) si falla.
    """
    # 1. Campos presentes
    for campo in CAMPOS_REQUERIDOS:
        if campo not in data:
            return f"Falta el campo requerido: '{campo}'", 400

    # 2. Tipos numéricos
    for campo in CAMPOS_REQUERIDOS:
        if not isinstance(data[campo], (int, float)):
            return f"El campo '{campo}' debe ser numérico", 400

    # 3. Rangos válidos
    for campo, (minv, maxv) in RANGOS.items():
        val = data[campo]
        if not (minv <= val <= maxv):
            return f"'{campo}' debe estar entre {minv} y {maxv}, se recibió {val}", 400

    return None, None


# ─── GET / ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "mensaje": "API del Gestor de Tareas con IA funcionando",
        "modelo_cargado": modelo is not None,
        "endpoints": {
            "POST /predecir":     "Predice prioridad de una tarea",
            "GET  /modelos":      "Compara múltiples modelos ML",
            "GET  /tareas/stats": "Estadísticas del dataset",
        }
    }), 200


# ─── POST /predecir ───────────────────────────────────────────────────────────
@app.route("/predecir", methods=["POST"])
def predecir():
    # 1. Verificar que el modelo está listo
    if modelo is None:
        return jsonify({
            "error": "Modelo no disponible. Ejecuta modelo.py para entrenarlo."
        }), 503

    # 2. Parsear JSON del body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Body inválido. Se esperaba JSON."}), 400

    # 3. Validar campos y rangos
    error_msg, status_code = validar_campos(data)
    if error_msg:
        return jsonify({"error": error_msg}), status_code

    # 4. Crear DataFrame con el mismo orden de features del entrenamiento
    try:
        entrada = pd.DataFrame([{
            "urgencia":     data["urgencia"],
            "dificultad":   data["dificultad"],
            "tiempo":       data["tiempo"],
            "fecha_limite": data["fecha_limite"],
            "impacto":      data["impacto"],
            "tipo":         data["tipo"],
        }])

        # 5. Predicción
        pred  = int(modelo.predict(entrada)[0])
        proba = modelo.predict_proba(entrada)[0]

        return jsonify({
            "prioritaria": pred,                          # 0 = normal, 1 = alta prioridad
            "confianza":   round(float(proba[pred]), 4),  # confianza de la clase predicha
            "probabilidades": {
                "normal":      round(float(proba[0]), 4),
                "prioritaria": round(float(proba[1]), 4),
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500


# ─── GET /modelos ─────────────────────────────────────────────────────────────
@app.route("/modelos", methods=["GET"])
def comparar_modelos():
    """
    Entrena y compara múltiples modelos en tiempo real.
    Retorna métricas de validación cruzada para mostrar en el frontend.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        df = pd.read_csv("dataset_tareas.csv")
        X  = df[["urgencia", "dificultad", "tiempo", "fecha_limite", "impacto", "tipo"]]
        y  = df["prioritaria"]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        candidatos = {
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42,
            ),
            "SVM": SVC(
                kernel="rbf", probability=True, random_state=42,
            ),
        }

        # Agregar LightGBM si está instalado
        try:
            import lightgbm as lgb
            candidatos["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=42, verbose=-1,
            )
        except ImportError:
            pass

        resultados = []
        for nombre, clf in candidatos.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", clf),
            ])

            acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

            resultados.append({
                "modelo":        nombre,
                "accuracy_mean": round(acc_scores.mean() * 100, 2),
                "accuracy_std":  round(acc_scores.std()  * 100, 2),
                "auc_mean":      round(auc_scores.mean(), 4),
            })

        # Ordenar de mayor a menor accuracy
        resultados.sort(key=lambda r: r["accuracy_mean"], reverse=True)
        resultados[0]["mejor"] = True  # marcar el ganador

        return jsonify({"modelos": resultados}), 200

    except FileNotFoundError:
        return jsonify({"error": "dataset_tareas.csv no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── GET /tareas/stats ────────────────────────────────────────────────────────
@app.route("/tareas/stats", methods=["GET"])
def estadisticas():
    """Estadísticas descriptivas del dataset para el dashboard del frontend."""
    try:
        df = pd.read_csv("dataset_tareas.csv")

        total        = len(df)
        prioritarias = int(df["prioritaria"].sum())
        normales     = total - prioritarias

        return jsonify({
            "total":             total,
            "prioritarias":      prioritarias,
            "normales":          normales,
            "pct_prioritarias":  round(prioritarias / total * 100, 2),
            "promedios": {
                "urgencia":     round(df["urgencia"].mean(), 2),
                "impacto":      round(df["impacto"].mean(), 2),
                "dificultad":   round(df["dificultad"].mean(), 2),
                "tiempo":       round(df["tiempo"].mean(), 2),
                "fecha_limite": round(df["fecha_limite"].mean(), 2),
            },
            "distribucion_tipo": df["tipo"].value_counts().to_dict(),
        }), 200

    except FileNotFoundError:
        return jsonify({"error": "dataset_tareas.csv no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Ejecutar servidor ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)