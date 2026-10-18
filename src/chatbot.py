
from __future__ import annotations

import re
import sys

import requests

MAX_REINTENTOS = 3     
MAX_RECHAZOS_API = 3   

CONTACTO_HUMANO = "Te paso con una persona del equipo de acopio. Indícale que usaste el asistente; ella revisará el grano."

SLOTS = [
    ("Area", "el área del grano", "los píxeles que ocupa el grano en la imagen", "12000"),
    ("Perimeter", "el perímetro del grano", "el largo de su contorno, en píxeles", "450"),
    ("Major_Axis_Length", "el largo del grano (eje mayor)", "la distancia más larga, en píxeles", "190"),
    ("Minor_Axis_Length", "el ancho del grano (eje menor)", "la distancia más corta, en píxeles", "86"),
    ("Convex_Area", "el área convexa", "el área del contorno 'estirado' alrededor del grano, en píxeles", "12300"),
    ("Extent", "la razón de ocupación (extent)", "un número entre 0 y 1: grano / rectángulo que lo encierra", "0.66"),
]
NOMBRE = {s[0]: s[1] for s in SLOTS}

PALABRAS_HUMANO = ("humano", "persona", "asesor", "operador", "agente", "ayudante real")
PALABRAS_CANCELAR = ("cancelar", "salir", "adios", "adiós", "chau", "terminar", "fin")
PALABRAS_REINICIAR = ("reiniciar", "empezar de nuevo", "otra vez", "de nuevo", "borrar todo")
PALABRAS_AYUDA = ("ayuda", "help", "no entiendo", "como funciona", "cómo funciona", "?")
SI = ("si", "sí", "claro", "ok", "dale", "yes", "s", "vale", "empezar", "comenzar", "listo")
NO = ("no", "n", "nop", "nada")
SALUDOS = ("hola", "buenas", "buenos dias", "buenos días", "buen dia", "buen día", "hey", "inicio", "start")


class ClienteHTTP:
    """Cliente mínimo de la API; devuelve (status, json). Lanza ConnectionError si no responde."""

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
    t = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", t)  # "12 500" -> "12500"
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if len(nums) != 1:
        return None
    return float(nums[0])


class Chatbot:
    def __init__(self, cliente):
        self.cliente = cliente
        self.reiniciar_todo()
        self._desempeno = None

    def reiniciar_todo(self) -> None:
        self.estado = "INICIO"
        self.datos: dict[str, float] = {}
        self.fallos = 0
        self.rechazos_api = 0
        self.terminado = False
        self.ultimo_resultado = None  

    def _pendiente(self) -> str | None:
        return next((s[0] for s in SLOTS if s[0] not in self.datos), None)

    def _pregunta(self, clave: str) -> str:
        _, nombre, expl, ejemplo = next(s for s in SLOTS if s[0] == clave)
        n = [s[0] for s in SLOTS].index(clave) + 1
        return f"({n}/{len(SLOTS)}) Dime {nombre}: {expl}. Por ejemplo: {ejemplo}."

    def _derivar(self, motivo: str) -> str:
        self.estado = "DERIVADO"
        self.terminado = True
        self.datos.clear()
        return f"{motivo} {CONTACTO_HUMANO}"

    def saludo(self) -> str:
        return ("¡Hola! Soy el asistente de acopio de arroz. Con las medidas de un grano te digo si es "
                "probablemente de la variedad Cammeo u Osmancik.\n"
                "No guardo tus datos: solo los uso en esta consulta. Escribe 'ayuda' si te pierdes, "
                "'reiniciar' para empezar de nuevo o 'persona' para hablar con alguien.\n"
                "¿Empezamos? (sí / no)")

    def _ayuda(self) -> str:
        base = ("Te iré pidiendo 6 medidas del grano, una por una (números, con punto o coma decimal). "
                "Puedes escribir 'reiniciar', 'cancelar' o 'persona'. ")
        if self.estado == "PIDIENDO" and self._pendiente():
            return base + "\n" + self._pregunta(self._pendiente())
        return base + "¿Empezamos? (sí / no)"

    def procesar(self, texto: str) -> str:
        if self.terminado:
            return "La conversación terminó. Escribe 'hola' para iniciar una nueva consulta."
        t = _norm(texto or "")

        if not t:
            return self._no_entendi("No recibí ningún mensaje.")
        if _tiene_datos_personales(t):
            return ("Por tu privacidad no necesito datos personales (correo, teléfono...). "
                    "Escribe solo la medida del grano." + self._reprompt())
        if _contiene(t, PALABRAS_HUMANO):
            return self._derivar("Claro.")
        if _contiene(t, PALABRAS_CANCELAR):
            self.datos.clear(); self.terminado = True
            return "Listo, cancelé la consulta y no guardé nada. ¡Hasta pronto!"
        if _contiene(t, PALABRAS_REINICIAR):
            self.datos.clear(); self.fallos = 0; self.rechazos_api = 0; self.estado = "PIDIENDO"
            return "Empecemos de nuevo, borré las medidas anteriores.\n" + self._pregunta(SLOTS[0][0])
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
            return self._derivar("No logro entenderte y no quiero hacerte perder tiempo.")
        quedan = MAX_REINTENTOS - self.fallos
        return f"{prefijo} Intentemos otra vez ({quedan} intento{'s' if quedan != 1 else ''} más antes de pasarte con una persona).{self._reprompt()}"

    def _en_inicio(self, t: str) -> str:
        if t.strip("!¡.,") in SALUDOS:
            return self.saludo()
        if _contiene(t, SI):
            self.estado = "PIDIENDO"; self.fallos = 0
            return "Perfecto. " + self._pregunta(SLOTS[0][0])
        if _contiene(t, NO):
            self.terminado = True
            return "Está bien, no hay problema. ¡Hasta pronto!"
        return self._no_entendi("No te entendí: responde 'sí' para empezar o 'no' para salir.")

    def _en_pidiendo(self, texto_original: str) -> str:
        clave = self._pendiente()
        valor = extraer_numero(texto_original)
        if valor is None:
            return self._no_entendi(f"No encontré un único número en tu respuesta. Necesito {NOMBRE[clave]} como número.")
        if valor <= 0:
            return self._no_entendi(f"{NOMBRE[clave].capitalize()} debe ser mayor que 0.")
        if clave == "Extent" and valor > 1:
            return self._no_entendi("Ese dato es una razón entre 0 y 1 (por ejemplo 0.66).")
        self.datos[clave] = valor
        self.fallos = 0
        sig = self._pendiente()
        if sig:
            return f"Anotado: {NOMBRE[clave]} = {valor:g}.\n" + self._pregunta(sig)
        return self._consultar()

    def _consultar(self) -> str:
        try:
            status, resp = self.cliente.predecir(dict(self.datos))
        except ConnectionError:
            return self._derivar("Ahora mismo no puedo conectarme con el sistema de clasificación.")
        if status == 400:
            self.rechazos_api += 1
            detalles = resp.get("detalles", [])
            if self.rechazos_api >= MAX_RECHAZOS_API:
                return self._derivar("El sistema rechazó varias veces las medidas.")
            malos = {c for d in detalles for c in re.findall(r"'(\w+)'", d) if c in self.datos}
            for c in malos or list(self.datos):
                self.datos.pop(c, None)
            primero = self._pendiente()
            return ("El sistema detectó un problema con las medidas: " + " ".join(detalles) +
                    f"\nVamos a corregirlo.\n{self._pregunta(primero)}")
        if status != 200:
            return self._derivar("El sistema tuvo un problema inesperado.")
        return self._formatear(resp)

    def _formatear(self, r: dict) -> str:
        self.ultimo_resultado = dict(r)
        clase, prob = r["clase"], round(r["probabilidad"] * 100)
        msg = f"Resultado: probablemente es arroz {clase} (seguridad estimada: {prob} %)."
        if r.get("confianza") == "baja":
            msg += ("\nOjo: el caso está cerca del límite entre las dos variedades. Te recomiendo medir el grano "
                    "de nuevo o pedir que una persona lo revise antes de decidir.")
        else:
            msg += "\nLa forma del grano apunta con claridad a esa variedad."
        msg += ("\nRecuerda: es una estimación basada solo en la forma del grano; "
                "no reemplaza un análisis de laboratorio" + self._texto_desempeno() + ".")
        self.datos.clear()  # privacidad: se borran las medidas
        self.estado = "OTRO"
        self.fallos = 0
        return msg + "\n¿Quieres consultar otro grano? (sí / no)"

    def _texto_desempeno(self) -> str:
        try:
            if self._desempeno is None:
                st, info = self.cliente.info()
                self._desempeno = info["desempeno"]["accuracy_validacion"] if st == 200 else False
        except (ConnectionError, KeyError):
            self._desempeno = False
        if self._desempeno:
            return f"; en nuestras pruebas acierta cerca de {round(self._desempeno * 100)} de cada 100 granos, así que a veces se equivoca"
        return ""

    def _en_otro(self, t: str) -> str:
        if _contiene(t, SI):
            self.estado = "PIDIENDO"; self.fallos = 0; self.rechazos_api = 0
            return "Muy bien, otro grano.\n" + self._pregunta(SLOTS[0][0])
        if _contiene(t, NO):
            self.terminado = True
            return "Gracias por usar el asistente. ¡Hasta pronto!"
        return self._no_entendi("Responde 'sí' para otro grano o 'no' para terminar.")


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


if __name__ == "__main__": 
    main()
