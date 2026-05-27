import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "urgencia": np.random.randint(1, 11, n),
    "dificultad": np.random.randint(1, 11, n),
    "tiempo": np.random.randint(1, 11, n),
    "fecha_limite": np.random.randint(1, 11, n),
    "impacto": np.random.randint(1, 11, n),
    "tipo": np.random.choice([0,1,2], n)
})

# Score realista
score = (
    df["urgencia"] * 0.3 +
    df["impacto"] * 0.25 +
    (10 - df["tiempo"]) * 0.15 +
    (10 - df["fecha_limite"]) * 0.2 +
    df["dificultad"] * 0.1
)

# Ruido
ruido = np.random.normal(0, 2, n)

score_final = score + ruido

# Variable objetivo
df["prioritaria"] = (score_final > 6.5).astype(int)

# Guardar dataset
df.to_csv("dataset_tareas.csv", index=False)

print("Dataset mejorado creado")