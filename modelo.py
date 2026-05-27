# modelo.py mejorado
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("dataset_tareas.csv")

X = df[["urgencia", "dificultad", "tiempo", "fecha_limite", "impacto", "tipo"]]
y = df["prioritaria"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: escala + modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(pipeline, X, y, cv=5)
print("CV promedio:", scores.mean())

with open("modelo_entrenado.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(" Pipeline XGBoost guardado")