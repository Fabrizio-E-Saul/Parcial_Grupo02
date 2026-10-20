
from __future__ import annotations

import re
import sys

import requests

from .config import SLOTS, TEXTOS as T, contactos_whatsapp
from .preprocesamiento_api import incoherencias

MAX_REINTENTOS = 3     
MAX_RECHAZOS_API = 3   

NOMBRE = {s[0]: s[1] for s in SLOTS}

PALABRAS_HUMANO = ("humano", "persona", "asesor", "operador", "agente", "ayudante real")
PALABRAS_CANCELAR = ("cancelar", "salir", "adios", "adiós", "chau", "terminar", "fin")
PALABRAS_REINICIAR = ("reiniciar", "empezar de nuevo", "otra vez", "de nuevo", "borrar todo")
PALABRAS_AYUDA = ("ayuda", "help", "no entiendo", "como funciona", "cómo funciona", "?")
SI = ("si", "sí", "claro", "ok", "dale", "yes", "s", "vale", "empezar", "comenzar", "listo")
NO = ("no", "n", "nop", "nada")
SALUDOS = ("hola", "buenas", "buenos dias", "buenos días", "buen dia", "buen día", "hey", "inicio", "start")


class ClienteHTTP:

    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 5.0):
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def _req(self, metodo, ruta, **kw):
        try:
            r = requests.request(metodo, self.base + ruta, timeout=self.timeout, **kw)
            return r.status_code, r.json()
        except (requests.RequestException, ValueError) as e:
            raise ConnectionError(str(e)) from e

    def info(self):
        return self._req("GET", "/info")

    def predecir(self, payload: dict):
        return self._req("POST", "/predecir", json=payload)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _contiene(t: str, palabras) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", t) for p in palabras if p != "?") or ("?" in palabras and "?" in t)


def _tiene_datos_personales(t: str) -> bool:
    return "@" in t or bool(re.search(r"\d[\d\s\-]{7,}\d", t)) and len(re.findall(r"\d", t)) >= 9


def extraer_numero(texto: str) -> float | None:
    """Saca UN número del texto (acepta coma decimal, 'px', espacios). None si no hay o hay varios."""
    t = texto.strip().lower().replace(",", ".")
    t = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", t)  
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if len(nums) != 1:
        return None
    return float(nums[0])


def _g(x: float) -> str:
    return f"{x:g}"


class Chatbot:
    def __init__(self, cliente):
        self.cliente = cliente
        self._info_cache = None
        self.reiniciar_todo()

    def reiniciar_todo(self) -> None:
        self.estado = "INICIO"
        self.datos: dict[str, float] = {}
        self.fallos = 0
        self.rechazos_api = 0
        self.terminado = False
        self.ultimo_resultado = None  
        self.contactos: list[dict] = []   

    def _info(self) -> dict:
        if self._info_cache is None:
            try:
                st, info = self.cliente.info()
                self._info_cache = info if st == 200 and isinstance(info, dict) else {}
            except ConnectionError:
                self._info_cache = {}
        return self._info_cache

    def _rangos(self) -> dict:
        return self._info().get("entrada", {}).get("rangos_dataset") or {}

    def limites(self, clave: str):
        """(mínimo, máximo) permitidos ahora para esa medida; None si no se conocen.
        Además del rango del dataset aplica las relaciones: área convexa >= área y ancho <= largo."""
        lo, hi = self._rangos().get(clave) or (None, None)
        if clave == "Convex_Area" and "Area" in self.datos:
            lo = max(lo, self.datos["Area"]) if lo is not None else self.datos["Area"]
        if clave == "Minor_Axis_Length" and "Major_Axis_Length" in self.datos:
            hi = min(hi, self.datos["Major_Axis_Length"]) if hi is not None else self.datos["Major_Axis_Length"]
        return lo, hi

    def limites_pendiente(self):
        """Datos de la medida que se está pidiendo (la interfaz web los usa para validar al teclear)."""
        clave = self._pendiente() if self.estado == "PIDIENDO" else None
        if not clave:
            return None
        lo, hi = self.limites(clave)
        return {"clave": clave, "nombre": NOMBRE[clave], "min": lo, "max": hi}

    def _pendiente(self) -> str | None:
        return next((s[0] for s in SLOTS if s[0] not in self.datos), None)

    def _pregunta(self, clave: str) -> str:
        _, nombre, expl, ejemplo = next(s for s in SLOTS if s[0] == clave)
        n = [s[0] for s in SLOTS].index(clave) + 1
        lo, hi = self._rangos().get(clave) or (None, None)
        rango = T["rango"].format(lo=_g(lo), hi=_g(hi)) if lo is not None else ""
        return T["pregunta"].format(n=n, total=len(SLOTS), nombre=nombre, expl=expl, ejemplo=ejemplo, rango=rango)

    def _derivar(self, motivo: str) -> str:
        self.estado = "DERIVADO"
        self.terminado = True
        self.datos.clear()
        self.contactos = contactos_whatsapp()
        return f"{motivo} {T['derivar']}"

    def saludo(self) -> str:
        return T["saludo"]

    def _ayuda(self) -> str:
        if self.estado == "PIDIENDO" and self._pendiente():
            return T["ayuda_base"] + "\n" + self._pregunta(self._pendiente())
        return T["ayuda_base"] + " " + T["empezamos"]

    def procesar(self, texto: str) -> str:
        if self.terminado:
            return T["terminada"]
        t = _norm(texto or "")

        if not t:
            return self._no_entendi(T["sin_mensaje"])
        if _tiene_datos_personales(t):
            return T["privacidad"] + self._reprompt()
        # comandos globales
        if _contiene(t, PALABRAS_HUMANO):
            return self._derivar("Claro.")
        if _contiene(t, PALABRAS_CANCELAR):
            self.datos.clear(); self.terminado = True
            return T["cancelar"]
        if _contiene(t, PALABRAS_REINICIAR):
            self.datos.clear(); self.fallos = 0; self.rechazos_api = 0; self.estado = "PIDIENDO"
            return T["reiniciar"] + "\n" + self._pregunta(SLOTS[0][0])
        if _contiene(t, PALABRAS_AYUDA) and self.estado != "PIDIENDO":
            return self._ayuda()
        if self.estado == "PIDIENDO" and _contiene(t, ("ayuda", "no entiendo", "help")):
            return self._ayuda()

        if self.estado == "INICIO":
            return self._en_inicio(t)
        if self.estado == "PIDIENDO":
            return self._en_pidiendo(texto)
        if self.estado == "OTRO":
            return self._en_otro(t)
        return self._no_entendi("No pude continuar.")

    def _reprompt(self) -> str:
        if self.estado == "PIDIENDO" and self._pendiente():
            return "\n" + self._pregunta(self._pendiente())
        return ""

    def _no_entendi(self, prefijo: str) -> str:
        self.fallos += 1
        if self.fallos >= MAX_REINTENTOS:
            return self._derivar(T["ilegible"])
        quedan = MAX_REINTENTOS - self.fallos
        extra = T["intentos"].format(quedan=quedan, s="s" if quedan != 1 else "")
        return f"{prefijo} {extra}{self._reprompt()}"

    def _en_inicio(self, t: str) -> str:
        if t.strip("!¡.,") in SALUDOS:
            return self.saludo()
        if _contiene(t, SI):
            self.estado = "PIDIENDO"; self.fallos = 0
            return self._pregunta(SLOTS[0][0])
        if _contiene(t, NO):
            self.terminado = True
            return T["sin_inicio"]
        return self._no_entendi(T["no_entendi"])

    def _en_pidiendo(self, texto_original: str) -> str:
        clave = self._pendiente()
        nombre = NOMBRE[clave]
        valor = extraer_numero(texto_original)
        if valor is None:
            return self._no_entendi(T["no_numero"].format(nombre=nombre))
        if valor <= 0:
            return self._no_entendi(T["mayor_cero"].format(nombre=nombre.capitalize()))
        if clave == "Minor_Axis_Length" and valor > self.datos["Major_Axis_Length"]:
            return self._no_entendi(T["ancho_mayor"].format(largo=_g(self.datos["Major_Axis_Length"])))
        if clave == "Convex_Area" and valor < self.datos["Area"]:
            return self._no_entendi(T["convexa_menor"].format(area=_g(self.datos["Area"])))
        if clave == "Minor_Axis_Length":
            malas = incoherencias(self.datos["Area"], self.datos["Perimeter"], self.datos["Major_Axis_Length"], valor)
            if malas:
                for k in ("Area", "Perimeter", "Major_Axis_Length"):
                    self.datos.pop(k, None)          
                return self._no_entendi(T["incoherente"].format(detalles=" ".join(self._amigable(m)[0].upper() + self._amigable(m)[1:] for m in malas)))
        lo, hi = self._rangos().get(clave) or (None, None)
        if lo is not None and not (lo <= valor <= hi):
            return self._no_entendi(T["fuera_rango"].format(nombre=nombre.capitalize(), lo=_g(lo), hi=_g(hi)))
        if lo is None and clave == "Extent" and valor > 1:
            return self._no_entendi(T["razon"])
        self.datos[clave] = valor
        self.fallos = 0
        sig = self._pendiente()
        if sig:
            return self._pregunta(sig)
        return self._consultar()

    def _amigable(self, texto: str) -> str:
        """Cambia 'Convex_Area' por 'el área convexa' en los mensajes de la API."""
        return re.sub(r"'(\w+)'", lambda m: NOMBRE.get(m.group(1), m.group(0)), texto)

    def _consultar(self) -> str:
        try:
            status, resp = self.cliente.predecir(dict(self.datos))
        except ConnectionError:
            return self._derivar(T["sin_conexion"])
        if status == 400:
            self.rechazos_api += 1
            detalles = resp.get("detalles", [])
            if self.rechazos_api >= MAX_RECHAZOS_API:
                return self._derivar(T["api_rechaza"])
            malos = {c for d in detalles for c in re.findall(r"'(\w+)'", d) if c in self.datos}
            for c in malos or list(self.datos):
                self.datos.pop(c, None)
            primero = self._pendiente()
            texto = T["problema_medidas"].format(detalles=" ".join(self._amigable(d) for d in detalles))
            return f"{texto}\n{self._pregunta(primero)}"
        if status != 200:
            return self._derivar(T["api_error"])
        return self._formatear(resp)

    def _formatear(self, r: dict) -> str:
        self.ultimo_resultado = dict(r)
        msg = T["resultado"].format(clase=r["clase"], prob=round(r["probabilidad"] * 100))
        if r.get("confianza") == "baja":
            msg += T["cerca_limite"]
        msg += "\n" + T["aviso_estimacion"].format(desempeno=self._texto_desempeno())
        self.datos.clear()  # privacidad: se borran las medidas
        self.estado = "OTRO"
        self.fallos = 0
        return msg + "\n" + T["otro_grano"]

    def _texto_desempeno(self) -> str:
        acc = self._info().get("desempeno", {}).get("accuracy_validacion")
        return T["desempeno"].format(n=round(acc * 100)) if acc else ""

    def _en_otro(self, t: str) -> str:
        if _contiene(t, SI):
            self.estado = "PIDIENDO"; self.fallos = 0; self.rechazos_api = 0
            return T["otro_si"] + "\n" + self._pregunta(SLOTS[0][0])
        if _contiene(t, NO):
            self.terminado = True
            return T["otro_no"]
        return self._no_entendi(T["no_entendi"])


def main(argv=None) -> None:  
    url = (argv or sys.argv[1:] or ["http://127.0.0.1:5000"])[0]
    bot = Chatbot(ClienteHTTP(url))
    print("Asistente:", bot.saludo())
    while not bot.terminado:
        try:
            entrada = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        print("Asistente:", bot.procesar(entrada))
    for c in bot.contactos:
        print(f"  WhatsApp {c['nombre']}: {c['url']}")


if __name__ == "__main__":  
    main()
