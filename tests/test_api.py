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


def test_info_publica_los_rangos_del_dataset_completo(cliente_api):
    r = cliente_api.get("/info").json["entrada"]["rangos_dataset"]
    assert r["Area"] == [7551.0, 18913.0] and r["Extent"][0] == 0.497


@pytest.mark.parametrize("cambio", [
    {"Area": 7550, "Convex_Area": 7800},                 
    {"Area": 18914, "Convex_Area": 19050},              
    {"Perimeter": 600}, {"Major_Axis_Length": 140, "Minor_Axis_Length": 80},
    {"Minor_Axis_Length": 108}, {"Extent": 0.49}, {"Extent": 0.9},
])
def test_valores_fuera_del_minimo_o_maximo_del_dataset_dan_400(cliente_api, grano, cambio):
    r = cliente_api.post("/predecir", json={**grano, **cambio})
    assert r.status_code == 400 and "fuera del rango del dataset" in " ".join(r.json["detalles"])


def test_ningun_grano_real_del_dataset_es_rechazado():
    """Los límites son estrictos pero no deben dejar fuera a ningún grano observado (extremos incluidos)."""
    from src import datos as D
    df, _ = D.limpiar(D.cargar_crudo())
    rangos = D.rangos_dataset()
    for fila in df[D.COLUMNAS].to_dict("records"):
        validar_entrada(fila, rangos)


def test_convex_area_menor_que_area_y_largo_menor_que_ancho_dan_400(cliente_api, grano):
    a = cliente_api.post("/predecir", json={**grano, "Area": 14000, "Convex_Area": 13000})
    b = cliente_api.post("/predecir", json={**grano, "Major_Axis_Length": 150, "Minor_Axis_Length": 160})
    assert a.status_code == 400 and "Convex_Area" in " ".join(a.json["detalles"])
    assert b.status_code == 400 and "ejes" in " ".join(b.json["detalles"])


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


CASO_IMPOSIBLE = {"Area": 15271, "Perimeter": 454, "Major_Axis_Length": 151.264,
                  "Minor_Axis_Length": 59.532, "Convex_Area": 16371, "Extent": 0.68}


def test_grano_imposible_aunque_cada_valor_este_en_rango_da_400_y_explica(cliente_api):
    r = cliente_api.post("/predecir", json=CASO_IMPOSIBLE)
    texto = " ".join(r.json["detalles"])
    assert r.status_code == 400 and "no es coherente" in texto
    assert "entre 6365 y 7426" in texto and "entre 330 y 399" in texto     


def test_la_coherencia_acepta_los_ejemplos_de_la_interfaz(cliente_api):
    for g in ({"Area": 15500, "Perimeter": 520, "Major_Axis_Length": 225, "Minor_Axis_Length": 88,
               "Convex_Area": 15900, "Extent": 0.62},
              {"Area": 10500, "Perimeter": 410, "Major_Axis_Length": 165, "Minor_Axis_Length": 82,
               "Convex_Area": 10700, "Extent": 0.70}):
        assert cliente_api.post("/predecir", json=g).status_code == 200
