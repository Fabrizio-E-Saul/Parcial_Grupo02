import math

import pytest

from src.preprocesamiento_api import ErrorValidacion, validar_entrada

CAMPOS = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Convex_Area", "Extent"]


def test_salud(cliente_api):
    r = cliente_api.get("/salud")
    assert r.status_code == 200 and r.json["estado"] == "ok" and r.json["modelo_cargado"] is True


def test_prediccion_exitosa_devuelve_clase_probabilidad_y_version(cliente_api, grano):
    r = cliente_api.post("/predecir", json=grano)
    assert r.status_code == 200
    j = r.json
    assert j["clase"] in ("Cammeo", "Osmancik")
    assert 0.5 <= j["probabilidad"] <= 1 and 0 <= j["probabilidad_cammeo"] <= 1
    assert j["version"] and j["confianza"] in ("alta", "baja")


def test_grano_grande_es_cammeo_y_pequeno_es_osmancik(cliente_api):
    cammeo = {"Area": 15500, "Perimeter": 520, "Major_Axis_Length": 225, "Minor_Axis_Length": 88,
              "Convex_Area": 15900, "Extent": 0.62}
    osmancik = {"Area": 10500, "Perimeter": 410, "Major_Axis_Length": 165, "Minor_Axis_Length": 82,
                "Convex_Area": 10700, "Extent": 0.70}
    assert cliente_api.post("/predecir", json=cammeo).json["clase"] == "Cammeo"
    assert cliente_api.post("/predecir", json=osmancik).json["clase"] == "Osmancik"


def test_eccentricity_opcional_da_el_mismo_resultado(cliente_api, grano):
    ecc = math.sqrt(1 - (grano["Minor_Axis_Length"] / grano["Major_Axis_Length"]) ** 2)
    a = cliente_api.post("/predecir", json=grano).json
    b = cliente_api.post("/predecir", json={**grano, "Eccentricity": ecc}).json
    assert abs(a["probabilidad_cammeo"] - b["probabilidad_cammeo"]) < 1e-4


def test_acepta_numeros_como_texto_con_coma(cliente_api, grano):
    r = cliente_api.post("/predecir", json={**grano, "Extent": "0,65"})
    assert r.status_code == 200


def test_latencia_se_mide_y_es_razonable(cliente_api, grano):
    r = cliente_api.post("/predecir", json=grano)
    assert r.json["latencia_ms"] > 0 and r.json["latencia_ms"] < 500
    assert "X-Latencia-ms" in r.headers


def test_info_documenta_el_contrato(cliente_api):
    j = cliente_api.get("/info").json
    assert j["entrada"]["obligatorios"] == CAMPOS and "desempeno" in j and "salida" in j


@pytest.mark.parametrize("campo", CAMPOS)
def test_campo_faltante_da_400_y_nombra_el_campo(cliente_api, grano, campo):
    grano.pop(campo)
    r = cliente_api.post("/predecir", json=grano)
    assert r.status_code == 400 and campo in " ".join(r.json["detalles"])


@pytest.mark.parametrize("cambio, texto", [
    ({"Area": "abc"}, "número"),
    ({"Area": -5}, "mayor que 0"),
    ({"Area": 0}, "mayor que 0"),
    ({"Extent": 1.7}, "1"),
    ({"Minor_Axis_Length": 300}, "ejes"),
    ({"Convex_Area": 100}, "Convex_Area"),
    ({"Area": True}, "número"),
    ({"Area": None}, "número"),
    ({"Area": 9999999, "Convex_Area": 99999999}, "rango"),
    ({"Perimeter": 1}, "rango"),
    ({"Campo_Inventado": 3}, "no reconocidos"),
    ({"Eccentricity": 0.1}, "coherente"),
    ({"Eccentricity": 1.5}, "menor que 1"),
])
def test_entradas_invalidas_dan_400_con_mensaje_claro(cliente_api, grano, cambio, texto):
    r = cliente_api.post("/predecir", json={**grano, **cambio})
    assert r.status_code == 400
    assert r.json["error"] == "Entrada inválida."
    assert texto in " ".join(r.json["detalles"])


def test_json_roto_da_400(cliente_api):
    r = cliente_api.post("/predecir", data="esto no es json", content_type="application/json")
    assert r.status_code == 400 and "JSON" in r.json["error"]


def test_cuerpo_que_no_es_objeto_da_400(cliente_api):
    assert cliente_api.post("/predecir", json=[1, 2, 3]).status_code == 400


def test_ruta_inexistente_404_y_metodo_incorrecto_405(cliente_api):
    assert cliente_api.get("/no_existe").status_code == 404
    assert cliente_api.get("/predecir").status_code == 405


def test_validar_entrada_calcula_eccentricity(grano):
    v = validar_entrada(grano)
    esperado = math.sqrt(1 - (88 / 205) ** 2)
    assert abs(v["Eccentricity"] - esperado) < 1e-9 and len(v) == 7


def test_validar_entrada_junta_varios_errores():
    with pytest.raises(ErrorValidacion) as e:
        validar_entrada({"Area": "x"})
    assert len(e.value.mensajes) >= 2   
