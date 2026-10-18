"""Corre TODO el entrenamiento y deja los modelos, fichas y figuras en modelos/ y docs/.

Uso:  python entrenar.py
Es reproducible: con la misma semilla produce los mismos resultados. Cada ejecución
parte de cero (borra modelos/ anteriores), de modo que la prueba se mira una sola vez
por corrida.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src import datos as D
from src import evaluacion as E
from src import modelo as M

RAIZ = Path(__file__).resolve().parent
DIR_MODELOS = RAIZ / "modelos"
DIR_FIG = RAIZ / "docs" / "figuras"


def limpiar_salidas() -> None:
    if DIR_MODELOS.exists():
        shutil.rmtree(DIR_MODELOS)
    DIR_MODELOS.mkdir(parents=True)
    DIR_FIG.mkdir(parents=True, exist_ok=True)


def main(semilla: int = D.SEMILLA, con_figuras: bool = True) -> dict:
    t0 = time.time()
    limpiar_salidas()

    # ---- E2: datos
    p, calidad = D.cargar_particiones(semilla=semilla)
    fuga = D.comprobar_sin_fuga(p)
    assert sum(fuga.values()) == 0, f"Filas repetidas entre particiones: {fuga}"
    datos_info = {
        "archivo": "data/Rice_Cammeo_Osmancik.arff", "sha256": D.hash_archivo(),
        "calidad": calidad, "semilla_particion": semilla,
        "filas": {"entrenamiento": len(p["X_ent"]), "validacion": len(p["X_val"]), "prueba": len(p["X_pru"])},
    }
    rangos = {c: [float(p["X_ent"][c].min()), float(p["X_ent"][c].max())] for c in D.COLUMNAS}
    cortes_area = [float(v) for v in p["X_ent"]["Area"].quantile([1 / 3, 2 / 3])]

    # ---- E3/E4: entrenar versiones y evaluar EN VALIDACIÓN
    modelos, res_val, ruido = {}, {}, {}
    for nombre in M.NOMBRES:
        m = M.entrenar(nombre, p["X_ent"], p["y_ent"], semilla)
        proba_val = M.prob_cammeo(m, p["X_val"])
        umbral, _ = E.elegir_umbral(p["y_val"], proba_val)
        res_val[nombre] = E.metricas(p["y_val"], proba_val, umbral)
        res_val[nombre]["metricas_umbral_0.5"] = {
            k: v for k, v in E.metricas(p["y_val"], proba_val, 0.5).items() if k in ("accuracy", "f1_macro", "recall_cammeo")}
        cv = E.variabilidad_cv(nombre, p["X_ent"], p["y_ent"], semilla)
        sem = (E.variabilidad_semillas(nombre, p["X_ent"], p["y_ent"], p["X_val"], p["y_val"])
               if nombre.startswith("mlp") else None)
        ruido[nombre] = {"cv_entrenamiento": cv, "semillas_validacion": sem}
        modelos[nombre] = (m, umbral)
        print(f"[{nombre:20s}] umbral={umbral:.2f} costo_val={res_val[nombre]['costo_por_grano']:.4f} "
              f"F1m_val={res_val[nombre]['f1_macro']:.4f} CV F1m={cv['f1_macro_media']:.4f}±{cv['f1_macro_desv']:.4f}")

    # ---- ¿diferencias reales o ruido? McNemar en validación (umbral 0.5)
    pred05 = {n: (M.prob_cammeo(modelos[n][0], p["X_val"]) >= 0.5).astype(int) for n in M.NOMBRES}
    mc = {f"regresion_logistica_vs_{n}": E.mcnemar(p["y_val"], pred05["regresion_logistica"], pred05[n])
          for n in ("mlp_v1", "mlp_v2")}

    # ---- ganadora (solo validación) + fichas
    ganadora, motivo = E.elegir_ganadora(res_val)
    for nombre in M.NOMBRES:
        m, umbral = modelos[nombre]
        E.guardar_version(nombre, m, {
            "descripcion": nombre, "datos": datos_info,
            "parametros": {k: (list(v) if isinstance(v, tuple) else v) for k, v in M.parametros(nombre, semilla).items()},
            "preprocesamiento": "StandardScaler ajustado solo con entrenamiento (dentro del Pipeline)",
            "umbral_decision": umbral,
            "costos_supuestos": {"falso_negativo": E.COSTO_FN, "falso_positivo": E.COSTO_FP,
                                 "nota": "Supuesto del escenario; no son precios reales."},
            "rangos_entrada_entrenamiento": rangos, "cortes_area_terciles": cortes_area,
            "metricas_validacion": res_val[nombre], "variabilidad": ruido[nombre],
            "semilla": semilla, "promovible": nombre != "baseline_trivial",
        }, DIR_MODELOS)
    E.promover(ganadora, motivo, DIR_MODELOS)

    # ---- evaluación final UNICA en prueba (solo la promovida)
    m_g, u_g = modelos[ganadora]
    res_pru = E.evaluar_en_prueba(ganadora, m_g, p["X_pru"], p["y_pru"], u_g, DIR_MODELOS)
    err_pru = E.errores_por_grupo(m_g, p["X_pru"], p["y_pru"], u_g, cortes_area)
    err_val = E.errores_por_grupo(m_g, p["X_val"], p["y_val"], u_g, cortes_area)
    # añadir resultado de prueba a la ficha ganadora (después de elegir, no para elegir)
    ruta = DIR_MODELOS / f"{ganadora}.json"
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    ficha["metricas_prueba"] = res_pru
    ficha["errores_por_grupo_prueba"] = err_pru
    ruta.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")

    resumen = {
        "ganadora": ganadora, "motivo": motivo, "validacion": res_val, "ruido": ruido, "mcnemar_validacion": mc,
        "prueba": res_pru, "errores_prueba": err_pru, "errores_validacion": err_val,
        "calidad": calidad, "particion": datos_info["filas"], "rangos": rangos,
        "segundos": round(time.time() - t0, 1),
    }
    (DIR_MODELOS / "resumen_entrenamiento.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    if con_figuras:
        from src.figuras import generar_figuras
        generar_figuras(p, modelos, res_val, ganadora, DIR_FIG)
    print(f"\nGanadora: {ganadora}\n  {motivo}")
    print(f"Prueba (única mirada): F1m={res_pru['f1_macro']:.4f}  acc={res_pru['accuracy']:.4f}  "
          f"recall_cammeo={res_pru['recall_cammeo']:.4f}  costo={res_pru['costo_por_grano']:.4f}")
    print(f"Tiempo: {resumen['segundos']} s")
    return resumen


if __name__ == "__main__":
    main()
    sys.exit(0)
