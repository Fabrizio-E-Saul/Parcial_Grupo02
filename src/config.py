
from __future__ import annotations

import re
from urllib.parse import quote


INTEGRANTES = [
    {"nombre": "Sebastian", "whatsapp": "51901967965"},
    {"nombre": "Fabrizio", "whatsapp": "51901117357"},
    {"nombre": "Jocabed", "whatsapp": "51977775442"},
    {"nombre": "Vanessa", "whatsapp": "51926506042"},
]
MENSAJE_WHATSAPP = "Hola {nombre}, usé el asistente de arroz (Cammeo / Osmancik) y necesito ayuda con un grano."


def contactos_whatsapp() -> list[dict]:
    """Lista [{'nombre', 'url'}] con enlaces wa.me listos para abrir. Omite números no válidos."""
    salida = []
    for m in INTEGRANTES:
        numero = re.sub(r"\D", "", str(m.get("whatsapp", "")))
        if len(numero) < 10:      # vacío o con marcador de posición
            continue
        texto = quote(MENSAJE_WHATSAPP.format(nombre=m["nombre"]))
        salida.append({"nombre": m["nombre"], "url": f"https://wa.me/{numero}?text={texto}"})
    return salida


# ✏️ MEDIDAS QUE PIDE EL BOT -------------------------------------------------------------------
# (columna del dataset, nombre amigable, explicación corta, ejemplo)
SLOTS = [
    ("Area", "el área del grano", "los píxeles que ocupa", "12000"),
    ("Perimeter", "el perímetro del grano", "el largo de su contorno, en píxeles", "450"),
    ("Major_Axis_Length", "el largo del grano", "su distancia más larga, en píxeles", "190"),
    ("Minor_Axis_Length", "el ancho del grano", "su distancia más corta, en píxeles", "86"),
    ("Convex_Area", "el área convexa", "el área del contorno 'estirado' alrededor del grano", "12300"),
    ("Extent", "la razón de ocupación", "grano / rectángulo que lo encierra", "0.66"),
]

# ✏️ TEXTOS DEL CHATBOT ------------------------------------------------------------------------
# Las llaves entre { } se rellenan solas: no las borres.
TEXTOS = {
    "saludo": ("¡Hola! Te ayudo a saber si un grano es Cammeo u Osmancik. "
               "No guardo tus datos. ¿Empezamos? (sí / no)"),
    "pregunta": "({n}/{total}) Dime {nombre}: {expl}. Ej.: {ejemplo}.{rango}",
    "rango": " Rango: {lo} a {hi}.",
    "ayuda_base": "Te pido 6 medidas, una por una, solo con números. Con los botones puedes reiniciar o hablar con una persona.",
    "empezamos": "¿Empezamos? (sí / no)",
    "sin_inicio": "Está bien, ¡hasta pronto!",
    "otro_si": "Otro grano, entonces.",
    "otro_no": "Gracias por usar el asistente. ¡Hasta pronto!",
    "cancelar": "Listo, cancelé la consulta y no guardé nada. ¡Hasta pronto!",
    "reiniciar": "Empecemos de nuevo; borré las medidas anteriores.",
    "terminada": "La conversación terminó. Escribe 'hola' para iniciar una nueva consulta.",
    "privacidad": "Por tu privacidad no necesito datos personales. Escribe solo la medida del grano.",
    "derivar": "Te paso con una persona del equipo.",
    "no_entendi": "No te entendí: responde 'sí' o 'no'.",
    "no_numero": "Necesito {nombre} como número.",
    "mayor_cero": "{nombre} debe ser mayor que 0.",
    "razon": "Es una razón entre 0 y 1 (por ejemplo 0.66).",
    "fuera_rango": "{nombre} debe estar entre {lo} y {hi}.",
    "ancho_mayor": "El ancho no puede ser mayor que el largo ({largo}). Si el largo estaba mal, pulsa Reiniciar.",
    "convexa_menor": "El área convexa no puede ser menor que el área ({area}). Si el área estaba mal, pulsa Reiniciar.",
    "incoherente": "Esas medidas no pueden ser de un mismo grano: {detalles} Revisa las medidas y empecemos otra vez.",
    "intentos": "({quedan} intento{s} más antes de pasarte con una persona).",
    "sin_mensaje": "No recibí ningún mensaje.",
    "sin_conexion": "Ahora mismo no puedo conectarme con el sistema de clasificación.",
    "api_rechaza": "El sistema rechazó varias veces las medidas.",
    "api_error": "El sistema tuvo un problema inesperado.",
    "ilegible": "No logro entenderte y no quiero hacerte perder tiempo.",
    "problema_medidas": "Hay un problema con las medidas: {detalles}\nCorrijámoslo.",
    "resultado": "Es arroz {clase} (seguridad {prob} %).",
    "cerca_limite": " Está cerca del límite entre las dos variedades: mide de nuevo o pide que una persona lo revise.",
    "aviso_estimacion": "Es una estimación por la forma del grano; no reemplaza un análisis de laboratorio{desempeno}.",
    "desempeno": " (en pruebas acierta cerca de {n} de cada 100 granos)",
    "otro_grano": "¿Consultamos otro grano? (sí / no)",
}
