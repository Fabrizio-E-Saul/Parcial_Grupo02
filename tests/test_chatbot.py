import pytest

from src.chatbot import Chatbot, extraer_numero

MEDIDAS = ["14162", "487,5", "205", "88", "14494", "0.65"]


class ClienteFlask:

    def __init__(self, c):
        self.c = c

    def info(self):
        r = self.c.get("/info")
        return r.status_code, r.json

    def predecir(self, payload):
        r = self.c.post("/predecir", json=payload)
        return r.status_code, r.json


class ClienteFalso:

    def __init__(self, respuestas, info_ok=True):
        self.respuestas, self.llamadas, self.info_ok = list(respuestas), 0, info_ok

    def info(self):
        if not self.info_ok:
            raise ConnectionError("caído")
        return 200, {"desempeno": {"accuracy_validacion": 0.94}}

    def predecir(self, payload):
        self.llamadas += 1
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


OK_ALTA = (200, {"clase": "Cammeo", "probabilidad": 0.97, "probabilidad_cammeo": 0.97, "confianza": "alta"})
OK_BAJA = (200, {"clase": "Osmancik", "probabilidad": 0.55, "probabilidad_cammeo": 0.45, "confianza": "baja"})
R400 = (400, {"error": "Entrada inválida.", "detalles": ["'Minor_Axis_Length' no puede ser mayor que 'Major_Axis_Length'"]})


def _llenar(bot, medidas=MEDIDAS):
    bot.procesar("hola"); bot.procesar("sí")
    out = ""
    for m in medidas:
        out = bot.procesar(m)
    return out


def test_camino_feliz_con_la_api_real(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    out = _llenar(bot)
    assert "Cammeo" in out and "laboratorio" in out and "otro grano" in out
    assert bot.estado == "OTRO"


def test_saludo_explica_que_no_guarda_datos():
    assert "No guardo" in Chatbot(ClienteFalso([])).procesar("hola")


def test_las_medidas_se_borran_despues_del_resultado():
    bot = Chatbot(ClienteFalso([OK_ALTA]))
    _llenar(bot)
    assert bot.datos == {}


def test_segundo_grano_tras_responder_si():
    bot = Chatbot(ClienteFalso([OK_ALTA]))
    _llenar(bot)
    assert "(1/6)" in bot.procesar("sí")


def test_terminar_tras_el_resultado():
    bot = Chatbot(ClienteFalso([OK_ALTA]))
    _llenar(bot)
    assert "Hasta pronto" in bot.procesar("no") and bot.terminado


def test_resultado_de_baja_confianza_recomienda_verificar():
    bot = Chatbot(ClienteFalso([OK_BAJA]))
    out = _llenar(bot)
    assert "cerca del límite" in out and "persona" in out


def test_el_bot_reconoce_sus_limitaciones():
    out = _llenar(Chatbot(ClienteFalso([OK_ALTA])))
    assert "estimación" in out and "94 de cada 100" in out


def test_si_info_falla_igual_responde_sin_cifra():
    out = _llenar(Chatbot(ClienteFalso([OK_ALTA], info_ok=False)))
    assert "Cammeo" in out and "de cada 100" not in out


@pytest.mark.parametrize("texto, esperado", [
    ("12000", 12000.0), ("487,5", 487.5), ("  0.65 ", 0.65), ("son 14 162 px", 14162.0),
    ("unos 205 píxeles", 205.0), ("abc", None), ("10 y 20", None), ("", None),
])
def test_extraer_numero(texto, esperado):
    assert extraer_numero(texto) == esperado


def test_respuesta_no_numerica_pide_de_nuevo_la_misma_medida():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí"); bot.procesar("12000")
    out = bot.procesar("no sé cuánto es")
    assert "perímetro" in out and "intento" in out


def test_tres_respuestas_ilegibles_derivan_a_una_persona():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    bot.procesar("xx"); bot.procesar("yy")
    out = bot.procesar("zz")
    assert "persona" in out and bot.estado == "DERIVADO" and bot.terminado


def test_un_acierto_reinicia_el_contador_de_fallos():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    bot.procesar("xx"); bot.procesar("yy")
    bot.procesar("12000")          
    bot.procesar("xx"); bot.procesar("yy")
    assert bot.estado == "PIDIENDO"


def test_pedir_una_persona_deriva_en_cualquier_momento():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    assert "persona" in bot.procesar("quiero hablar con una persona") and bot.estado == "DERIVADO"


def test_cancelar_no_guarda_nada():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí"); bot.procesar("12000")
    out = bot.procesar("cancelar")
    assert "no guardé nada" in out and bot.datos == {} and bot.terminado


def test_reiniciar_borra_lo_ingresado():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí"); bot.procesar("12000")
    out = bot.procesar("reiniciar")
    assert "(1/6)" in out and bot.datos == {}


def test_ayuda_en_medio_de_la_recoleccion_repite_la_pregunta():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    assert "(1/6)" in bot.procesar("ayuda")


@pytest.mark.parametrize("texto", ["mi correo es ana@mail.com", "llámame al 987 654 321 00"])
def test_datos_personales_se_ignoran_y_se_avisa(texto):
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    out = bot.procesar(texto)
    assert "privacidad" in out and bot.datos == {}


def test_valor_fuera_de_rango_local_no_se_acepta():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    for m in ["12000", "450", "190", "86", "12300"]:
        bot.procesar(m)
    out = bot.procesar("5")          
    assert "entre 0 y 1" in out and "Extent" not in bot.datos


def test_rechazo_400_de_la_api_pide_corregir_los_datos_senalados():
    bot = Chatbot(ClienteFalso([R400, OK_ALTA]))
    out = _llenar(bot)
    assert "problema con las medidas" in out and "(3/6)" in out
    bot.procesar("205")
    assert "Cammeo" in bot.procesar("88")


def test_tres_rechazos_de_la_api_derivan():
    bot = Chatbot(ClienteFalso([R400, R400, R400]))
    _llenar(bot)                                   
    bot.procesar("205"); bot.procesar("88")        
    bot.procesar("205")
    out = bot.procesar("88")                       
    assert "persona" in out and bot.estado == "DERIVADO"


def test_api_caida_deriva_con_mensaje_claro():
    bot = Chatbot(ClienteFalso([ConnectionError("sin conexión")]))
    out = _llenar(bot)
    assert "no puedo conectarme" in out and bot.estado == "DERIVADO"


def test_mensaje_vacio_cuenta_como_fallo():
    bot = Chatbot(ClienteFalso([]))
    assert "No recibí" in bot.procesar("   ")


def test_conversacion_terminada_no_sigue_respondiendo_datos():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("no")  # en INICIO: sale
    assert bot.terminado and "terminó" in bot.procesar("12000")


def test_chatbot_con_api_real_rechaza_largo_fuera_del_rango_del_dataset(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    out = _llenar(bot, ["14162", "487", "88"])         
    assert "entre 145.264 y 239.011" in out and "Major_Axis_Length" not in bot.datos


def test_chatbot_con_api_real_el_area_convexa_no_puede_ser_menor_que_el_area(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    out = _llenar(bot, ["14000", "487", "205", "88", "10000"])   
    assert "no puede ser menor que el área" in out and "Convex_Area" not in bot.datos


def test_chatbot_con_api_real_muestra_el_rango_en_cada_pregunta(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    bot.procesar("hola")
    assert "Rango: 7551 a 18913" in bot.procesar("sí")


def test_chatbot_con_api_real_el_limite_exacto_del_dataset_se_acepta(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    bot.procesar("hola"); bot.procesar("sí")
    assert "(2/6)" in bot.procesar("7551") and bot.datos["Area"] == 7551.0
    assert "problema" in bot.procesar("18914") or "entre" in bot.procesar("18914")   

def test_ancho_mayor_que_el_largo_se_rechaza_sin_rangos():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    for m in ["12000", "450", "100"]:
        bot.procesar(m)
    out = bot.procesar("150")                            
    assert "no puede ser mayor que el largo" in out and "Minor_Axis_Length" not in bot.datos


def test_area_convexa_menor_que_el_area_se_rechaza_sin_rangos():
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola"); bot.procesar("sí")
    for m in ["12000", "450", "190", "86"]:
        bot.procesar(m)
    out = bot.procesar("11000")
    assert "no puede ser menor que el área" in out and "Convex_Area" not in bot.datos


def test_limites_pendiente_para_la_interfaz(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    assert bot.limites_pendiente() is None                
    bot.procesar("hola"); bot.procesar("sí")
    lim = bot.limites_pendiente()
    assert lim["clave"] == "Area" and lim["min"] == 7551.0 and lim["max"] == 18913.0
    for m in ["14162", "487", "205", "88"]:
        bot.procesar(m)
    assert bot.limites_pendiente()["min"] == 14162.0      


def test_derivar_a_una_persona_entrega_los_whatsapp_configurados(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "INTEGRANTES", [
        {"nombre": "Ana", "whatsapp": "51 987 654 321"}, {"nombre": "Sin número", "whatsapp": "51XXXXXXXXX"}])
    bot = Chatbot(ClienteFalso([]))
    bot.procesar("hola")
    bot.procesar("quiero hablar con una persona")
    assert [c["nombre"] for c in bot.contactos] == ["Ana"]
    assert bot.contactos[0]["url"].startswith("https://wa.me/51987654321?text=")


def test_las_respuestas_del_bot_son_cortas(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    textos = [bot.procesar("hola"), bot.procesar("sí")] + [bot.procesar(m) for m in MEDIDAS]
    assert all(len(t) < 260 for t in textos)


def test_grano_imposible_se_detecta_al_dar_el_ancho_y_se_vuelve_a_pedir(cliente_api):
    bot = Chatbot(ClienteFlask(cliente_api))
    out = _llenar(bot, ["15271", "454", "151.264", "59.532"])
    assert "no pueden ser de un mismo grano" in out and "entre 6365 y 7426" in out
    assert "(1/6)" in out and bot.datos == {}                  


def test_medidas_coherentes_no_disparan_la_revision_de_coherencia():
    bot = Chatbot(ClienteFalso([]))
    out = _llenar(bot, ["12000", "450", "190", "86"])
    assert "(5/6)" in out and "mismo grano" not in out
