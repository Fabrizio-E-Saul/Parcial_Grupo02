"""E3 - Preprocesador y modelos: línea base trivial, modelo simple y dos redes neuronales.

Todos los modelos son `Pipeline(StandardScaler -> clasificador)`. Al ajustar el pipeline
SOLO con X de entrenamiento, el escalado aprende media y desviación únicamente de
entrenamiento (sin fuga hacia validación o prueba).

Las redes son `sklearn.neural_network.MLPClassifier`. Regularización usada:
  * L2 (parámetro `alpha`) y parada temprana (`early_stopping`), que reserva una
    fracción INTERNA del entrenamiento; no toca validación ni prueba.
  * No se usa dropout ni pesos de clase: MLPClassifier no los ofrece, y el desbalance
    es leve (57 % / 43 %); se maneja con F1 macro, PR-AUC y umbral por costo.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Orden = de más simple a más complejo (se usa como desempate en la promoción).
NOMBRES = ["baseline_trivial", "regresion_logistica", "mlp_v1", "mlp_v2"]


def parametros(nombre: str, semilla: int) -> dict:
    """Hiperparámetros de cada versión (se guardan en la ficha JSON)."""
    if nombre == "baseline_trivial":
        return {"strategy": "prior"}
    if nombre == "regresion_logistica":
        return {"C": 1.0, "max_iter": 1000}
    if nombre == "mlp_v1":  # red pequeña, regularización suave
        return {
            "hidden_layer_sizes": (16,), "activation": "relu", "alpha": 1e-3,
            "learning_rate_init": 1e-3, "batch_size": 64, "max_iter": 500,
            "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 15,
            "random_state": semilla,
        }
    if nombre == "mlp_v2":  # red más profunda, más regularización
        return {
            "hidden_layer_sizes": (32, 16), "activation": "relu", "alpha": 1e-2,
            "learning_rate_init": 1e-3, "batch_size": 64, "max_iter": 500,
            "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 15,
            "random_state": semilla,
        }
    raise ValueError(f"Modelo desconocido: {nombre}")


def construir(nombre: str, semilla: int = 42) -> Pipeline:
    """Crea (sin entrenar) el pipeline de la versión pedida."""
    p = parametros(nombre, semilla)
    if nombre == "baseline_trivial":
        clf = DummyClassifier(**p)
    elif nombre == "regresion_logistica":
        clf = LogisticRegression(random_state=semilla, **p)
    else:
        clf = MLPClassifier(**p)
    return Pipeline([("escalador", StandardScaler()), ("clf", clf)])


def entrenar(nombre: str, X_ent, y_ent, semilla: int = 42) -> Pipeline:
    """Entrena la versión con SOLO datos de entrenamiento."""
    modelo = construir(nombre, semilla)
    modelo.fit(X_ent, y_ent)
    return modelo


def prob_cammeo(modelo: Pipeline, X) -> "np.ndarray":  # noqa: F821
    """Probabilidad de la clase positiva (Cammeo = 1)."""
    return modelo.predict_proba(X)[:, 1]
