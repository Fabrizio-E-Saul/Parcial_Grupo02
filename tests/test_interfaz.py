"""Pruebas de la interfaz web (página, endpoint /chat y sesiones)."""
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
    assert j["progreso"]["actual"] == 0          # las medidas se borran tras responder


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
    assert _chat(cliente_api, "13000", b).json["progreso"]["actual"] == 1   # b no ve el dato de a


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
    assert all("vacio" in l for l in usos)       # solo el texto fijo del estado vacío


def test_el_chat_informa_los_limites_de_la_medida_pendiente(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    assert _chat(cliente_api, "sí", sid).json["pendiente"]["clave"] == "Area"
    j = _chat(cliente_api, "14162", sid).json
    assert j["pendiente"]["clave"] == "Perimeter" and j["pendiente"]["min"] < j["pendiente"]["max"]


def test_borrar_historial_elimina_la_sesion(cliente_api):
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    _chat(cliente_api, "sí", sid); _chat(cliente_api, "14162", sid)
    assert cliente_api.delete(f"/chat/{sid}").json == {"borrado": True}
    j = _chat(cliente_api, "hola", sid).json            # la sesión ya no existe: empieza de cero
    assert j["progreso"]["actual"] == 0 and j["estado"] == "INICIO"


def test_borrar_una_sesion_inexistente_no_falla(cliente_api):
    assert cliente_api.delete("/chat/no-existe-123").status_code == 200


def test_contactos_whatsapp_solo_incluye_numeros_validos(cliente_api, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "INTEGRANTES", [
        {"nombre": "Ana", "whatsapp": "51987654321"}, {"nombre": "Pendiente", "whatsapp": "51XXXXXXXXX"}])
    lista = cliente_api.get("/contactos").json["contactos"]
    assert [c["nombre"] for c in lista] == ["Ana"] and lista[0]["url"].startswith("https://wa.me/51987654321")


def test_derivar_a_persona_en_el_chat_devuelve_los_contactos(cliente_api, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "INTEGRANTES", [{"nombre": "Ana", "whatsapp": "51987654321"}])
    sid = _chat(cliente_api, "__nuevo__").json["sesion"]
    j = _chat(cliente_api, "quiero hablar con una persona", sid).json
    assert j["terminado"] and j["contactos"][0]["nombre"] == "Ana"
    assert _chat(cliente_api, "__nuevo__", sid).json["contactos"] == []   # la nueva consulta parte limpia


def test_ruta_de_imagenes_responde_404_si_no_hay_archivo(cliente_api):
    assert cliente_api.get("/img/no_existe.jpg").status_code == 404
    assert cliente_api.get("/img/../api.py").status_code in (404, 308)    # no sale de la carpeta


def test_el_html_tiene_botones_de_reinicio_y_borrado_y_resena():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "src" / "web" / "index.html").read_text(encoding="utf-8")
    for clave in ('id="btn-nuevo"', 'id="btn-borrar"', 'class="panel acerca"', "--foto-grupo", "/contactos", "/img/logo_fis.jpg", "/img/logo_uncp.png", "/img/grupo.jpg", "/img/arroz_", "fotoArroz"):
        assert clave in html
