"""E2 - Carga, limpieza y partición del dataset Rice (Cammeo / Osmancik).

Decisiones (justificadas en el informe, sección 2):
  * Los datos vienen de un .arff; la clase llega como bytes y se decodifica.
  * Se verifican nulos, duplicados y coherencia geométrica; si algo falla
    se ELIMINA la fila y se deja constancia en el reporte de calidad (no se imputa).
  * La partición es estratificada y con semilla fija: 60 % entrenamiento,
    20 % validación, 20 % prueba. El preprocesamiento (escalado) NO se ajusta aquí:
    se ajusta dentro de la tubería solo con entrenamiento (ver modelo.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
RUTA_ARFF = RAIZ / "data" / "Rice_Cammeo_Osmancik.arff"

SEMILLA = 42
COLUMNAS = [
    "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
    "Eccentricity", "Convex_Area", "Extent",
]
OBJETIVO = "Class"
# Clase positiva = Cammeo (1). Osmancik = 0.
CLASES = {0: "Osmancik", 1: "Cammeo"}


def cargar_crudo(ruta: Path | str = RUTA_ARFF) -> pd.DataFrame:
    """Lee el .arff tal cual, decodificando la columna de clase."""
    datos, _ = arff.loadarff(str(ruta))
    df = pd.DataFrame(datos)
    if df[OBJETIVO].map(lambda v: isinstance(v, bytes)).any():
        df[OBJETIVO] = df[OBJETIVO].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return df


def hash_archivo(ruta: Path | str = RUTA_ARFF) -> str:
    """SHA-256 del archivo de datos (trazabilidad en las fichas JSON)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica las reglas de calidad y devuelve (df_limpio, reporte)."""
    reporte = {"filas_iniciales": int(len(df))}
    df = df.copy()

    reporte["nulos"] = int(df.isna().sum().sum())
    df = df.dropna()

    reporte["duplicados"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Coherencia geométrica: lo que un grano real no puede incumplir.
    malas = (
        (df[COLUMNAS] <= 0).any(axis=1)
        | (df["Minor_Axis_Length"] > df["Major_Axis_Length"])
        | (df["Convex_Area"] < df["Area"])
        | (df["Extent"] > 1)
        | (df["Eccentricity"] >= 1)
    )
    reporte["incoherentes"] = int(malas.sum())
    df = df[~malas]

    reporte["clases_invalidas"] = int((~df[OBJETIVO].isin(["Cammeo", "Osmancik"])).sum())
    df = df[df[OBJETIVO].isin(["Cammeo", "Osmancik"])]

    reporte["filas_finales"] = int(len(df))
    reporte["distribucion_clases"] = {k: int(v) for k, v in df[OBJETIVO].value_counts().items()}
    return df.reset_index(drop=True), reporte


def a_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (7 variables) e y (1 = Cammeo, 0 = Osmancik)."""
    X = df[COLUMNAS].astype(float)
    y = (df[OBJETIVO] == "Cammeo").astype(int)
    return X, y


def particionar(X: pd.DataFrame, y: pd.Series, semilla: int = SEMILLA) -> dict:
    """60/20/20 estratificado. Devuelve dict con X_ent, y_ent, X_val, y_val, X_pru, y_pru."""
    X_ent, X_tmp, y_ent, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=semilla
    )
    X_val, X_pru, y_val, y_pru = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=semilla
    )
    return {
        "X_ent": X_ent, "y_ent": y_ent,
        "X_val": X_val, "y_val": y_val,
        "X_pru": X_pru, "y_pru": y_pru,
    }


def cargar_particiones(ruta: Path | str = RUTA_ARFF, semilla: int = SEMILLA):
    """Atajo: carga -> limpia -> particiona. Devuelve (particiones, reporte_calidad)."""
    df, reporte = limpiar(cargar_crudo(ruta))
    X, y = a_xy(df)
    return particionar(X, y, semilla), reporte


def comprobar_sin_fuga(p: dict) -> dict:
    """Cuenta filas idénticas compartidas entre particiones (contaminación de la prueba)."""
    def claves(X):
        return set(map(tuple, np.round(X.values, 6)))
    ent, val, pru = claves(p["X_ent"]), claves(p["X_val"]), claves(p["X_pru"])
    return {"ent_val": len(ent & val), "ent_pru": len(ent & pru), "val_pru": len(val & pru)}
