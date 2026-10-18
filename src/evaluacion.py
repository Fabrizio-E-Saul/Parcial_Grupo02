"""E4 - Métricas, umbral por costo, variabilidad, versionado y promoción.

Reglas de integridad que este módulo hace cumplir:
  * La versión ganadora se elige SOLO con validación (`elegir_ganadora`).
  * El conjunto de prueba se evalúa UNA sola vez (`evaluar_en_prueba` se niega a repetirse).
  * Cada versión guarda una ficha JSON (fecha, datos, parámetros, métricas, umbral).
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from . import modelo as M

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "modelos"

# --- Escenario de costos (SUPUESTO del grupo, ajustable) -------------------------------
# Clase positiva = Cammeo. Falso negativo = un grano Cammeo se registra como Osmancik.
# Se asume que ese error cuesta el doble que el inverso. NO son precios reales.
COSTO_FN = 2.0
COSTO_FP = 1.0
TOLERANCIA_EMPATE = 0.01  # diferencia de costo por grano que se considera "empate"


# ------------------------------------------------------------------ métricas
def metricas(y, proba, umbral: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "umbral": float(umbral),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_cammeo": float(precision_score(y, pred, zero_division=0)),
        "recall_cammeo": float(recall_score(y, pred, zero_division=0)),
        "f1_cammeo": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "costo_por_grano": float((COSTO_FN * fn + COSTO_FP * fp) / len(y)),
        "matriz": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def costo_esperado(y, proba, umbral: float) -> float:
    y = np.asarray(y)
    pred = (np.asarray(proba) >= umbral).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return (COSTO_FN * fn + COSTO_FP * fp) / len(y)


def elegir_umbral(y_val, proba_val, malla=None) -> tuple[float, float]:
    """Umbral que minimiza el costo esperado EN VALIDACIÓN. Devuelve (umbral, costo)."""
    malla = np.round(np.arange(0.05, 0.951, 0.01), 2) if malla is None else malla
    costos = [costo_esperado(y_val, proba_val, u) for u in malla]
    i = int(np.argmin(costos))  # en empate gana el umbral más bajo
    return float(malla[i]), float(costos[i])


# --------------------------------------------------------- variabilidad / ruido
def variabilidad_cv(nombre: str, X_ent, y_ent, semilla: int = 42, k: int = 5, repeticiones: int = 2) -> dict:
    """Validación cruzada estratificada repetida SOLO sobre entrenamiento (umbral 0.5)."""
    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeticiones, random_state=semilla)
    f1s, accs = [], []
    for idx_a, idx_b in rkf.split(X_ent, y_ent):
        m = M.entrenar(nombre, X_ent.iloc[idx_a], y_ent.iloc[idx_a], semilla)
        pred = m.predict(X_ent.iloc[idx_b])
        f1s.append(f1_score(y_ent.iloc[idx_b], pred, average="macro"))
        accs.append(accuracy_score(y_ent.iloc[idx_b], pred))
    return {
        "f1_macro_media": float(np.mean(f1s)), "f1_macro_desv": float(np.std(f1s, ddof=1)),
        "accuracy_media": float(np.mean(accs)), "accuracy_desv": float(np.std(accs, ddof=1)),
        "n_particiones": len(f1s),
    }


def variabilidad_semillas(nombre: str, X_ent, y_ent, X_val, y_val, semillas=(0, 1, 2, 3, 4)) -> dict:
    """Reentrena con varias semillas y mide F1 macro en validación (umbral 0.5)."""
    vals = []
    for s in semillas:
        m = M.entrenar(nombre, X_ent, y_ent, s)
        vals.append(f1_score(y_val, m.predict(X_val), average="macro"))
    return {"f1_macro_media": float(np.mean(vals)), "f1_macro_desv": float(np.std(vals, ddof=1)),
            "semillas": list(semillas)}


def mcnemar(y, pred_a, pred_b) -> dict:
    """Prueba de McNemar exacta: ¿difieren A y B más de lo que explicaría el azar?"""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    solo_a = int(((a == y) & (b != y)).sum())  # A acierta, B falla
    solo_b = int(((a != y) & (b == y)).sum())
    n = solo_a + solo_b
    p = 1.0 if n == 0 else float(binomtest(solo_a, n, 0.5).pvalue)
    return {"a_acierta_b_falla": solo_a, "b_acierta_a_falla": solo_b, "p_valor": p}


# ----------------------------------------------------------- análisis de errores
def errores_por_grupo(modelo, X, y, umbral: float, cortes_area: list[float]) -> dict:
    """Error por clase y por tercil de tamaño (Area). Los cortes vienen de ENTRENAMIENTO."""
    proba = M.prob_cammeo(modelo, X)
    pred = (proba >= umbral).astype(int)
    y = np.asarray(y)
    out = {"por_clase": {}, "por_tamano": {}}
    for nombre, c in (("Osmancik", 0), ("Cammeo", 1)):
        mask = y == c
        out["por_clase"][nombre] = {"n": int(mask.sum()), "tasa_error": float((pred[mask] != y[mask]).mean())}
    area = X["Area"].values
    limites = [-np.inf, cortes_area[0], cortes_area[1], np.inf]
    for etiqueta, lo, hi in zip(("pequeño", "mediano", "grande"), limites[:-1], limites[1:]):
        mask = (area > lo) & (area <= hi)
        if mask.sum():
            out["por_tamano"][etiqueta] = {"n": int(mask.sum()), "tasa_error": float((pred[mask] != y[mask]).mean())}
    return out


# ------------------------------------------------------------ versionado
def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def guardar_version(nombre: str, modelo, ficha: dict, directorio: Path = DIR_MODELOS) -> Path:
    """Guarda el modelo (.joblib) y su ficha (.json). Devuelve la ruta de la ficha."""
    directorio.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, directorio / f"{nombre}.joblib")
    ficha = {
        "version": nombre,
        "fecha_utc": _ahora(),
        "artefacto": f"{nombre}.joblib",
        "entorno": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__},
        **ficha,
    }
    ruta = directorio / f"{nombre}.json"
    ruta.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def elegir_ganadora(resultados_val: dict) -> tuple[str, str]:
    """Elige usando SOLO validación. `resultados_val[nombre]['costo_por_grano']`.

    Regla (fijada antes de mirar resultados): menor costo por grano con umbral óptimo;
    los modelos a menos de TOLERANCIA_EMPATE del mejor se consideran empatados y gana
    el MÁS SIMPLE (orden de M.NOMBRES). La línea base trivial no es promovible.
    """
    candidatos = {n: r for n, r in resultados_val.items() if n != "baseline_trivial"}
    mejor = min(r["costo_por_grano"] for r in candidatos.values())
    empatados = [n for n, r in candidatos.items() if r["costo_por_grano"] <= mejor + TOLERANCIA_EMPATE]
    ganadora = sorted(empatados, key=M.NOMBRES.index)[0]
    motivo = (f"mejor costo en validación = {mejor:.4f}; empatados (<= +{TOLERANCIA_EMPATE}): "
              f"{', '.join(sorted(empatados, key=M.NOMBRES.index))}; se elige el más simple: {ganadora}")
    return ganadora, motivo


def promover(nombre: str, motivo: str, directorio: Path = DIR_MODELOS) -> Path:
    """Marca la versión promovida. La API lee `promovida.json`."""
    ficha = json.loads((directorio / f"{nombre}.json").read_text(encoding="utf-8"))
    ruta = directorio / "promovida.json"
    ruta.write_text(json.dumps({"version": nombre, "fecha_utc": _ahora(), "motivo": motivo,
                                "artefacto": ficha["artefacto"], "ficha": f"{nombre}.json"},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def evaluar_en_prueba(nombre: str, modelo, X_pru, y_pru, umbral: float, directorio: Path = DIR_MODELOS) -> dict:
    """Evaluación FINAL en prueba. Solo se permite una vez (deja `prueba_evaluada.json`)."""
    marca = directorio / "prueba_evaluada.json"
    if marca.exists():
        raise RuntimeError("El conjunto de prueba ya fue evaluado: no se permite una segunda mirada.")
    res = metricas(y_pru, M.prob_cammeo(modelo, X_pru), umbral)
    marca.write_text(json.dumps({"version": nombre, "fecha_utc": _ahora(), "metricas": res},
                                indent=2, ensure_ascii=False), encoding="utf-8")
    return res
