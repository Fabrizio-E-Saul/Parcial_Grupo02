
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

from . import modelo as M
from .config import contactos_whatsapp
from .datos import CLASES, COLUMNAS, rangos_dataset
from .chatbot import Chatbot, SLOTS
from .preprocesamiento_api import OBLIGATORIOS, UNIDADES, ErrorValidacion, a_dataframe, validar_entrada

RAIZ = Path(__file__).resolve().parents[1]
MARGEN_CONFIANZA = 0.15  
MAX_SESIONES = 300      
TTL_SESION = 1800        


def cargar_promovida(directorio: Path | str | None = None) -> dict:
    """Lee modelos/promovida.json y carga el artefacto y la ficha de esa versión."""
    d = Path(directorio) if directorio else RAIZ / "modelos"
    prom = json.loads((d / "promovida.json").read_text(encoding="utf-8"))
    ficha = json.loads((d / prom["ficha"]).read_text(encoding="utf-8"))
    return {"modelo": joblib.load(d / prom["artefacto"]), "ficha": ficha, "version": prom["version"]}


def crear_app(directorio_modelos: Path | str | None = None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    estado = cargar_promovida(directorio_modelos)  
    modelo, ficha, version = estado["modelo"], estado["ficha"], estado["version"]
    umbral = float(ficha["umbral_decision"])
    rangos_entrenamiento = ficha["rangos_entrada_entrenamiento"]
    try:
        rangos = rangos_dataset()          
    except Exception:                      
        rangos = rangos_entrenamiento
    mv = ficha["metricas_validacion"]
    desempeno = {"accuracy_validacion": round(mv["accuracy"], 4), "f1_macro_validacion": round(mv["f1_macro"], 4)}

    def error(msg: str, detalles=None, codigo: int = 400):
        return jsonify({"error": msg, "detalles": detalles or [msg]}), codigo

    @app.get("/salud")
    def salud():
        return jsonify({"estado": "ok", "modelo_cargado": True, "version": version})

    def info_dict() -> dict:
        return {
            "version": version, "umbral_decision": umbral, "desempeno": desempeno,
            "entrada": {"obligatorios": OBLIGATORIOS, "opcionales": ["Eccentricity"], "unidades": UNIDADES,
                        "rangos_entrenamiento": rangos_entrenamiento, "rangos_dataset": rangos},
            "salida": {"clase": list(CLASES.values()), "probabilidad": "prob. de la clase predicha",
                       "probabilidad_cammeo": "prob. de Cammeo", "confianza": "alta | baja",
                       "version": "versión del modelo", "latencia_ms": "tiempo de cómputo"},
        }

    def ejecutar(payload) -> tuple[int, dict]:
        """Lógica común de /predecir y del chat: valida y predice. Devuelve (status, json)."""
        t0 = time.perf_counter()
        try:
            valores = validar_entrada(payload, rangos)
        except ErrorValidacion as e:
            return 400, {"error": "Entrada inválida.", "detalles": e.mensajes}
        p_cammeo = float(M.prob_cammeo(modelo, a_dataframe(valores))[0])
        es_cammeo = p_cammeo >= umbral
        return 200, {
            "clase": CLASES[int(es_cammeo)],
            "probabilidad": round(p_cammeo if es_cammeo else 1 - p_cammeo, 4),
            "probabilidad_cammeo": round(p_cammeo, 4),
            "confianza": "baja" if abs(p_cammeo - umbral) < MARGEN_CONFIANZA else "alta",
            "version": version, "umbral": umbral,
            "latencia_ms": round((time.perf_counter() - t0) * 1000, 3),
        }

    @app.get("/info")
    def info():
        return jsonify(info_dict())

    @app.post("/predecir")
    def predecir():
        payload = request.get_json(silent=True)
        if payload is None:
            return error("El cuerpo debe ser un JSON válido (Content-Type: application/json).")
        status, resp = ejecutar(payload)
        r = jsonify(resp)
        r.status_code = status
        if status == 200:
            r.headers["X-Latencia-ms"] = str(resp["latencia_ms"])
        return r

    class ClienteInterno:
        def info(self):
            return 200, info_dict()

        def predecir(self, payload):
            return ejecutar(payload)

    sesiones: dict[str, dict] = {}   
    def _limpiar_sesiones():
        ahora = time.time()
        for sid in [k for k, v in sesiones.items() if ahora - v["t"] > TTL_SESION]:
            sesiones.pop(sid, None)
        while len(sesiones) > MAX_SESIONES:
            sesiones.pop(min(sesiones, key=lambda k: sesiones[k]["t"]), None)

    def _sugerencias(bot: Chatbot) -> list[str]:
        if bot.terminado:
            return ["Nueva consulta"]
        if bot.estado in ("INICIO", "OTRO"):
            return ["Sí", "No"]
        return ["Ayuda", "Reiniciar", "Hablar con una persona"]

    @app.get("/")
    def interfaz():
        return send_from_directory(RAIZ / "src" / "web", "index.html")

    @app.get("/img/<path:nombre>")
    def imagen(nombre):
        return send_from_directory(RAIZ / "src" / "web" / "img", nombre)

    @app.get("/contactos")
    def contactos():
        return jsonify({"contactos": contactos_whatsapp()})

    @app.delete("/chat/<sid>")
    def borrar_chat(sid):
        sesiones.pop(sid, None)
        return jsonify({"borrado": True})

    @app.post("/chat")
    def chat():
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or not isinstance(datos.get("mensaje"), str):
            return error("Envía un JSON con el campo 'mensaje' (texto).")
        _limpiar_sesiones()
        sid = datos.get("sesion") if isinstance(datos.get("sesion"), str) and 8 <= len(datos["sesion"]) <= 64 else uuid.uuid4().hex
        mensaje = datos["mensaje"][:300]
        if mensaje == "__nuevo__" or sid not in sesiones:
            sesiones[sid] = {"bot": Chatbot(ClienteInterno()), "t": time.time()}
            bot = sesiones[sid]["bot"]
            texto = bot.saludo()
        else:
            bot = sesiones[sid]["bot"]
            sesiones[sid]["t"] = time.time()
            bot.ultimo_resultado = None
            texto = bot.procesar(mensaje)
        return jsonify({
            "sesion": sid, "respuesta": texto, "estado": bot.estado, "terminado": bot.terminado,
            "progreso": {"actual": len(bot.datos), "total": len(SLOTS)},
            "resultado": getattr(bot, "ultimo_resultado", None),
            "sugerencias": _sugerencias(bot),
            "pendiente": bot.limites_pendiente(),
            "contactos": bot.contactos,
        })

    @app.errorhandler(404)
    def no_encontrado(_):
        return error("Ruta no encontrada.", codigo=404)

    @app.errorhandler(405)
    def metodo(_):
        return error("Método no permitido para esta ruta.", codigo=405)

    return app


if __name__ == "__main__":  
    crear_app().run(host="127.0.0.1", port=5000)
