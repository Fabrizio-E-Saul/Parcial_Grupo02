import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src import evaluacion as E
from src import modelo as M


def test_mismo_seed_mismas_probabilidades(particiones):
    p, _ = particiones
    a = M.prob_cammeo(M.entrenar("mlp_v1", p["X_ent"], p["y_ent"], 42), p["X_val"])
    b = M.prob_cammeo(M.entrenar("mlp_v1", p["X_ent"], p["y_ent"], 42), p["X_val"])
    assert np.allclose(a, b)


def test_linea_base_trivial_no_supera_al_modelo_simple(particiones):
    p, _ = particiones
    f1 = {}
    for n in ("baseline_trivial", "regresion_logistica"):
        m = M.entrenar(n, p["X_ent"], p["y_ent"], 42)
        f1[n] = E.metricas(p["y_val"], M.prob_cammeo(m, p["X_val"]), 0.5)["f1_macro"]
    assert f1["regresion_logistica"] > f1["baseline_trivial"] + 0.3


def test_modelo_desconocido_lanza_error():
    with pytest.raises(ValueError):
        M.construir("modelo_que_no_existe")


def test_umbral_baja_si_el_falso_negativo_cuesta_mas():
    y = np.array([0] * 50 + [1] * 50)
    rng = np.random.default_rng(0)
    proba = np.clip(np.r_[rng.normal(.35, .15, 50), rng.normal(.65, .15, 50)], 0, 1)
    u_actual, _ = E.elegir_umbral(y, proba)
    E.COSTO_FN = 10.0
    try:
        u_caro, _ = E.elegir_umbral(y, proba)
    finally:
        E.COSTO_FN = 2.0
    assert u_caro <= u_actual


def test_metricas_en_caso_perfecto():
    m = E.metricas([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["accuracy"] == 1 and m["f1_macro"] == 1 and m["matriz"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_ganadora_empate_elige_el_mas_simple():
    res = {"baseline_trivial": {"costo_por_grano": 0.5}, "regresion_logistica": {"costo_por_grano": 0.090},
           "mlp_v1": {"costo_por_grano": 0.085}, "mlp_v2": {"costo_por_grano": 0.095}}
    g, _ = E.elegir_ganadora(res)
    assert g == "regresion_logistica"          


def test_ganadora_clara_gana_la_red():
    res = {"baseline_trivial": {"costo_por_grano": 0.5}, "regresion_logistica": {"costo_por_grano": 0.20},
           "mlp_v1": {"costo_por_grano": 0.08}, "mlp_v2": {"costo_por_grano": 0.15}}
    assert E.elegir_ganadora(res)[0] == "mlp_v1"


def test_la_linea_base_nunca_se_promueve():
    res = {"baseline_trivial": {"costo_por_grano": 0.0}, "regresion_logistica": {"costo_por_grano": 0.3},
           "mlp_v1": {"costo_por_grano": 0.3}, "mlp_v2": {"costo_por_grano": 0.3}}
    assert E.elegir_ganadora(res)[0] != "baseline_trivial"


def test_ficha_y_promocion_en_directorio_temporal(particiones):
    p, _ = particiones
    m = M.entrenar("regresion_logistica", p["X_ent"], p["y_ent"], 1)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        E.guardar_version("regresion_logistica", m, {"parametros": {"C": 1.0}, "umbral_decision": 0.5}, d)
        E.promover("regresion_logistica", "prueba", d)
        ficha = json.loads((d / "regresion_logistica.json").read_text(encoding="utf-8"))
        assert {"version", "fecha_utc", "parametros", "entorno"} <= set(ficha)
        assert json.loads((d / "promovida.json").read_text(encoding="utf-8"))["version"] == "regresion_logistica"


def test_la_prueba_solo_se_puede_evaluar_una_vez(particiones):
    p, _ = particiones
    m = M.entrenar("regresion_logistica", p["X_ent"], p["y_ent"], 1)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        E.evaluar_en_prueba("regresion_logistica", m, p["X_pru"], p["y_pru"], 0.5, d)
        with pytest.raises(RuntimeError):
            E.evaluar_en_prueba("regresion_logistica", m, p["X_pru"], p["y_pru"], 0.5, d)


def test_fichas_reales_tienen_trazabilidad_completa(directorio_modelos):
    for n in M.NOMBRES:
        f = json.loads((directorio_modelos / f"{n}.json").read_text(encoding="utf-8"))
        assert f["datos"]["sha256"] and f["fecha_utc"] and f["parametros"] is not None
        assert "metricas_validacion" in f and "umbral_decision" in f


def test_prueba_evaluada_una_sola_vez_en_la_corrida_real(directorio_modelos):
    marca = json.loads((directorio_modelos / "prueba_evaluada.json").read_text(encoding="utf-8"))
    prom = json.loads((directorio_modelos / "promovida.json").read_text(encoding="utf-8"))
    assert marca["version"] == prom["version"]
