"""
analisis.py - Comparativa y evaluación de modelos de clasificación
Gestor de Tareas con IA

Modelos evaluados:
  - XGBoost       :
  - LightGBM
  - Random Forest :
  - SVM

Ejecutar: python analisis.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ─── Intentar importar LightGBM (opcional) ───────────────────────────────────
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM no instalado. Instalar con: pip install lightgbm\n")

# ─── 1. Cargar datos ──────────────────────────────────────────────────────────
print("=" * 60)
print("  ANÁLISIS Y COMPARATIVA DE MODELOS - GESTOR DE TAREAS IA")
print("=" * 60)

df = pd.read_csv("dataset_tareas.csv")

print(f"\nDataset cargado: {len(df)} registros, {df.shape[1]} columnas")
print(f"   Distribución de clases:")
print(f"   - Prioritaria (1): {df['prioritaria'].sum()} ({df['prioritaria'].mean()*100:.1f}%)")
print(f"   - Normal     (0): {(~df['prioritaria'].astype(bool)).sum()} ({(1-df['prioritaria'].mean())*100:.1f}%)")

# ─── 2. Features y target (las mismas 6 del modelo en producción) ─────────────
FEATURES = ["urgencia", "dificultad", "tiempo", "fecha_limite", "impacto", "tipo"]

X = df[FEATURES]
y = df["prioritaria"]

# ─── 3. Split estratificado ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # mantiene proporción de clases en ambos sets
)

print(f"\nSplit: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ─── 4. Definir candidatos de modelos ────────────────────────────────────────
candidatos = {
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        probability=True,  # necesario para predict_proba y ROC-AUC
        random_state=42,
    ),
}

if LGBM_AVAILABLE:
    candidatos["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )

# ─── 5. Validación cruzada + evaluación en test ───────────────────────────────
print("\n" + "─" * 60)
print("  RESULTADOS POR MODELO")
print("─" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

for nombre, clf in candidatos.items():
    # Pipeline: siempre escalar antes del modelo
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", clf),
    ])

    # Validación cruzada (5 folds)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    # Entrenar en train completo y evaluar en test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc_test = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)

    resultados.append({
        "nombre":   nombre,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "acc_test": acc_test,
        "auc":      auc,
        "pipeline": pipeline,
        "y_pred":   y_pred,
    })

    print(f"\n  {nombre}")
    print(f"   CV Accuracy : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"   Test Accuracy: {acc_test*100:.2f}%")
    print(f"   ROC-AUC      : {auc:.4f}")

# ─── 6. Elegir el mejor modelo por CV ────────────────────────────────────────
mejor = max(resultados, key=lambda r: r["cv_mean"])

print("\n" + "─" * 60)
print(f"     MEJOR MODELO: {mejor['nombre']}")
print(f"     CV Accuracy : {mejor['cv_mean']*100:.2f}%")
print(f"     Test Accuracy: {mejor['acc_test']*100:.2f}%")
print(f"     ROC-AUC      : {mejor['auc']:.4f}")
print("─" * 60)

# ─── 7. Reporte detallado del mejor modelo ───────────────────────────────────
print(f"\nReporte de clasificación — {mejor['nombre']}:")
print(classification_report(
    y_test,
    mejor["y_pred"],
    target_names=["Normal (0)", "Prioritaria (1)"]
))

print(" Matriz de confusión:")
cm = confusion_matrix(y_test, mejor["y_pred"])
print(f"   Verdadero Negativo : {cm[0][0]}")
print(f"   Falso Positivo     : {cm[0][1]}")
print(f"   Falso Negativo     : {cm[1][0]}")
print(f"   Verdadero Positivo : {cm[1][1]}")

# ─── 8. Tabla resumen comparativa ────────────────────────────────────────────
print("\n Tabla resumen comparativa:")
print(f"{'Modelo':<16} {'CV Accuracy':>12} {'Test Accuracy':>14} {'ROC-AUC':>9}")
print("-" * 55)
for r in sorted(resultados, key=lambda x: x["cv_mean"], reverse=True):
    marca = " ← mejor" if r["nombre"] == mejor["nombre"] else ""
    print(
        f"{r['nombre']:<16}"
        f" {r['cv_mean']*100:>10.2f}%"
        f" {r['acc_test']*100:>12.2f}%"
        f" {r['auc']:>9.4f}"
        f"{marca}"
    )

print("\n Análisis completado.")
print(f"   Ejecuta modelo.py para guardar '{mejor['nombre']}' en producción.")