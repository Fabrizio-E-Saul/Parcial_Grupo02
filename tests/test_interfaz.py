import pytest


def _chat(cliente, mensaje, sesion=None):
    cuerpo = {"mensaje": mensaje}
    if sesion:
        cuerpo["sesion"] = sesion
    return cliente.post("/chat", json=cuerpo)


def test_la_pagina_principal_se_sirve(cliente_api):
    r = cliente_api.get("/")
    assert r.status_code == 200 and b"Asistente de acopio de arroz" in r.data


def test_chat_nuevo_devuelve_saludo_con_privacidad_y_sugerencias(cliente_api):
    r = _chat(cliente_api, "__nuevo__")
    assert r.status_code == 200
    j = r.json
    assert "No guardo" in j["respuesta"] and j["sugerencias"] == ["Sí", "No"] and len(j["sesion"]) >= 8
    assert j["progreso"] == {"actual": 0, "total": 6}


def test_conversacion_completa_por_la_interfaz_devuelve_resultado_estructurado(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    _chat(cliente_api, "sí", sid)
    j = None
    for m in ["14162", "487,5", "205", "88", "14494", "0.65"]:
        j = _chat(cliente_api, m, sid).json
    assert j["resultado"]["clase"] == "Cammeo" and j["estado"] == "OTRO"
    assert j["progreso"]["actual"] == 0          


def test_el_progreso_avanza_con_cada_medida(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    _chat(cliente_api, "sí", sid)
    assert _chat(cliente_api, "14162", sid).json["progreso"]["actual"] == 1
    assert _chat(cliente_api, "487", sid).json["progreso"]["actual"] == 2


def test_las_sesiones_son_independientes(cliente_api):
    a = _chat(cliente_api, "__nuevo__").json["sesion"]
    b = _chat(cliente_api, "__nuevo__").json["sesion"]
    _chat(cliente_api, "sí", a); _chat(cliente_api, "14162", a)
    _chat(cliente_api, "sí", b)
    assert _chat(cliente_api, "487", b).json["progreso"]["actual"] == 1   


def test_nueva_consulta_reinicia_la_sesion(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    _chat(cliente_api, "sí", sid); _chat(cliente_api, "14162", sid)
    j = _chat(cliente_api, "__nuevo__", sid).json
    assert j["progreso"]["actual"] == 0 and j["estado"] == "INICIO"


def test_derivar_a_persona_deja_solo_nueva_consulta(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    j = _chat(cliente_api, "quiero hablar con una persona", sid).json
    assert j["terminado"] and j["sugerencias"] == ["Nueva consulta"]


@pytest.mark.parametrize("cuerpo", [{}, {"mensaje": 5}, {"sesion": "abc"}])
def test_chat_con_cuerpo_invalido_da_400(cliente_api, cuerpo):
    assert cliente_api.post("/chat", json=cuerpo).status_code == 400


def test_sesion_desconocida_empieza_una_conversacion_nueva(cliente_api):
    j = _chat(cliente_api, "hola", "sesion-que-no-existe-123").json
    assert "No guardo" in j["respuesta"]


def test_el_html_no_inserta_texto_del_usuario_con_innerhtml():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src" / "web" / "index.html").read_text(encoding="utf-8")
    usos = [l for l in html.splitlines() if "innerHTML" in l]
    assert all("vacio" in l for l in usos)     
