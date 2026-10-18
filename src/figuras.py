"""Figuras del informe (se generan desde entrenar.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from . import datos as D
from . import evaluacion as E
from . import modelo as M

COL = {"Cammeo": "#c0392b", "Osmancik": "#2471a3"}


def generar_figuras(p, modelos, res_val, ganadora, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    X, y = p["X_ent"], p["y_ent"]

    # 1) distribución por clase y variable
    fig, ejes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, c in zip(ejes.ravel(), D.COLUMNAS):
        for k, nombre in D.CLASES.items():
            ax.hist(X.loc[y == k, c], bins=30, alpha=.55, label=nombre, color=COL[nombre])
        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=8)
    ejes.ravel()[-1].axis("off")
    ejes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Distribución de cada variable por clase (entrenamiento)")
    fig.tight_layout(); fig.savefig(destino / "fig1_distribuciones.png", dpi=140); plt.close(fig)

    # 2) correlaciones
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = X.corr().values
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(D.COLUMNAS, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(D.COLUMNAS, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046); ax.set_title("Correlación (entrenamiento)")
    fig.tight_layout(); fig.savefig(destino / "fig2_correlacion.png", dpi=140); plt.close(fig)

    # 3) comparación de modelos en validación
    nombres = M.NOMBRES
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.bar(nombres, [res_val[n]["f1_macro"] for n in nombres], color=["#95a5a6", "#2980b9", "#8e44ad", "#16a085"])
    a1.set_ylim(0, 1); a1.set_title("F1 macro en validación (umbral por costo)"); a1.tick_params(axis="x", labelsize=7, rotation=15)
    a2.bar(nombres, [res_val[n]["costo_por_grano"] for n in nombres], color=["#95a5a6", "#2980b9", "#8e44ad", "#16a085"])
    a2.set_title("Costo esperado por grano en validación"); a2.tick_params(axis="x", labelsize=7, rotation=15)
    fig.tight_layout(); fig.savefig(destino / "fig3_comparacion_modelos.png", dpi=140); plt.close(fig)

    # 4) ROC y PR en validación
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for n in nombres[1:]:
        pr = M.prob_cammeo(modelos[n][0], p["X_val"])
        RocCurveDisplay.from_predictions(p["y_val"], pr, name=n, ax=a1)
        PrecisionRecallDisplay.from_predictions(p["y_val"], pr, name=n, ax=a2)
    a1.set_title("ROC (validación)"); a2.set_title("Precisión-recall (validación)")
    fig.tight_layout(); fig.savefig(destino / "fig4_roc_pr.png", dpi=140); plt.close(fig)

    # 5) costo vs umbral (ganadora, validación) y matriz de confusión
    m, umbral = modelos[ganadora]
    pv = M.prob_cammeo(m, p["X_val"])
    malla = np.arange(0.05, 0.951, 0.01)
    costos = [E.costo_esperado(p["y_val"], pv, u) for u in malla]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(malla, costos); a1.axvline(umbral, color="r", ls="--", label=f"umbral elegido = {umbral:.2f}")
    a1.set_xlabel("umbral"); a1.set_ylabel("costo por grano"); a1.set_title(f"Costo vs. umbral ({ganadora}, validación)"); a1.legend()
    mt = E.metricas(p["y_val"], pv, umbral)["matriz"]
    mat = np.array([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]])
    a2.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a2.text(j, i, mat[i, j], ha="center", va="center", fontsize=14)
    a2.set_xticks([0, 1]); a2.set_yticks([0, 1]); a2.set_xticklabels(["Osmancik", "Cammeo"]); a2.set_yticklabels(["Osmancik", "Cammeo"])
    a2.set_xlabel("predicho"); a2.set_ylabel("real"); a2.set_title("Matriz de confusión (validación)")
    fig.tight_layout(); fig.savefig(destino / "fig5_umbral_confusion.png", dpi=140); plt.close(fig)

    # 6) calibración
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for n in nombres[1:]:
        fr, mp = calibration_curve(p["y_val"], M.prob_cammeo(modelos[n][0], p["X_val"]), n_bins=8)
        ax.plot(mp, fr, marker="o", label=n)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("probabilidad predicha"); ax.set_ylabel("frecuencia observada")
    ax.set_title("Calibración (validación)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(destino / "fig6_calibracion.png", dpi=140); plt.close(fig)
