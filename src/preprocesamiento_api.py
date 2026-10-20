"""E5 - Validación de entradas de la API (y su conversión a DataFrame para el modelo).

Contrato de entrada (JSON): 6 medidas obligatorias + Eccentricity opcional.
Si falta Eccentricity se calcula como sqrt(1 - (menor/mayor)^2), que es exactamente
cómo se define en el dataset (error máximo 6e-8 en los 3 810 granos).
Todos los mensajes de error están en español claro; la API los devuelve con código 400.
"""
from __future__ import annotations

import math

import pandas as pd

from .datos import COLUMNAS

OBLIGATORIOS = [c for c in COLUMNAS if c != "Eccentricity"]

UNIDADES = {
    "Area": "píxeles", "Perimeter": "píxeles", "Major_Axis_Length": "píxeles",
    "Minor_Axis_Length": "píxeles", "Convex_Area": "píxeles", "Extent": "razón 0-1",
    "Eccentricity": "0-1 (se calcula sola si no se envía)",
}


# Un grano real es casi una elipse: su área y su perímetro quedan cerca de los de la elipse formada
# por su largo y ancho. En el dataset: área/elipse entre 0.965 y 0.998; perímetro/elipse entre 0.992 y 1.089.
# Se aceptan márgenes algo más amplios para no rechazar granos reales.
MARGEN_AREA = (0.90, 1.05)
MARGEN_PERIMETRO = (0.95, 1.15)


def incoherencias(area: float, perimetro: float, mayor: float, menor: float) -> list[str]:
    """Mensajes (vacío si todo es coherente) cuando área o perímetro no cuadran con largo y ancho.
    Los usan la API y el chatbot, así que ambos explican lo mismo."""
    msgs = []
    elipse = math.pi / 4 * mayor * menor
    lo, hi = elipse * MARGEN_AREA[0], elipse * MARGEN_AREA[1]
    if not (lo <= area <= hi):
        msgs.append(f"'Area' ({area:g}) no es coherente con 'Major_Axis_Length' ({mayor:g}) y 'Minor_Axis_Length' "
                    f"({menor:g}): con esos ejes el área debería estar entre {lo:.0f} y {hi:.0f}.")
    a, b = mayor / 2, menor / 2
    per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    lo, hi = per * MARGEN_PERIMETRO[0], per * MARGEN_PERIMETRO[1]
    if not (lo <= perimetro <= hi):
        msgs.append(f"'Perimeter' ({perimetro:g}) no es coherente con 'Major_Axis_Length' ({mayor:g}) y "
                    f"'Minor_Axis_Length' ({menor:g}): con esos ejes el perímetro debería estar entre {lo:.0f} y {hi:.0f}.")
    return msgs


class ErrorValidacion(ValueError):
    """Entrada inválida. `.mensajes` es la lista de problemas encontrados."""

    def __init__(self, mensajes: list[str]):
        super().__init__("; ".join(mensajes))
        self.mensajes = mensajes


def _a_numero(nombre: str, valor, errores: list[str]):
    if isinstance(valor, bool) or valor is None:
        errores.append(f"'{nombre}' debe ser un número, no {valor!r}.")
        return None
    if isinstance(valor, str):
        try:
            valor = float(valor.strip().replace(",", "."))
        except ValueError:
            errores.append(f"'{nombre}' debe ser un número, pero llegó el texto {valor!r}.")
            return None
    if not isinstance(valor, (int, float)):
        errores.append(f"'{nombre}' debe ser un número (llegó un valor de tipo {type(valor).__name__}).")
        return None
    valor = float(valor)
    if math.isnan(valor) or math.isinf(valor):
        errores.append(f"'{nombre}' no puede ser NaN ni infinito.")
        return None
    return valor


def validar_entrada(payload, rangos: dict | None = None) -> dict:
    """Devuelve un dict {columna: float} con las 7 variables o lanza ErrorValidacion."""
    if not isinstance(payload, dict):
        raise ErrorValidacion(["El cuerpo debe ser un objeto JSON con las medidas del grano."])

    errores: list[str] = []
    desconocidos = sorted(set(payload) - set(COLUMNAS))
    if desconocidos:
        errores.append(f"Campos no reconocidos: {', '.join(desconocidos)}. Campos válidos: {', '.join(COLUMNAS)}.")

    faltan = [c for c in OBLIGATORIOS if c not in payload]
    if faltan:
        errores.append(f"Faltan campos obligatorios: {', '.join(faltan)}.")

    valores: dict[str, float] = {}
    for c in COLUMNAS:
        if c in payload:
            v = _a_numero(c, payload[c], errores)
            if v is not None:
                valores[c] = v
    if errores:
        raise ErrorValidacion(errores)

    for c, v in valores.items():
        if v <= 0:
            errores.append(f"'{c}' debe ser mayor que 0 (llegó {v}).")
    if "Extent" in valores and valores["Extent"] > 1:
        errores.append(f"'Extent' es una razón y no puede superar 1 (llegó {valores['Extent']}).")
    if "Eccentricity" in valores and valores["Eccentricity"] >= 1:
        errores.append(f"'Eccentricity' debe ser menor que 1 (llegó {valores['Eccentricity']}).")
    if errores:
        raise ErrorValidacion(errores)

    if valores["Minor_Axis_Length"] > valores["Major_Axis_Length"]:
        errores.append("'Minor_Axis_Length' no puede ser mayor que 'Major_Axis_Length': ¿se invirtieron los ejes?")
    if valores["Convex_Area"] < valores["Area"]:
        errores.append("'Convex_Area' no puede ser menor que 'Area' (el casco convexo contiene al grano).")
    if errores:
        raise ErrorValidacion(errores)

    ecc_calc = math.sqrt(1 - (valores["Minor_Axis_Length"] / valores["Major_Axis_Length"]) ** 2)
    ecc_enviada = "Eccentricity" in valores
    if not ecc_enviada:
        valores["Eccentricity"] = ecc_calc
    elif abs(valores["Eccentricity"] - ecc_calc) > 0.01:
        errores.append(f"'Eccentricity' ({valores['Eccentricity']:.3f}) no es coherente con los ejes "
                       f"(deberían dar {ecc_calc:.3f}).")

    if rangos:
        # Estricto: cada valor debe estar entre el mínimo y el máximo observados en el dataset.
        for c, v in valores.items():
            if c == "Eccentricity" and not ecc_enviada:
                continue  # la calculamos nosotros a partir de los ejes (ya validados)
            lo, hi = rangos[c]
            if not (lo <= v <= hi):
                unidad = " píxeles" if UNIDADES[c] == "píxeles" else ""
                errores.append(f"'{c}' = {v:g} está fuera del rango del dataset ({lo:g} a {hi:g}{unidad}).")
    if errores:
        raise ErrorValidacion(errores)

    # Coherencia entre medidas: no se aceptan combinaciones que no existen en un grano real.
    errores.extend(incoherencias(valores["Area"], valores["Perimeter"],
                                 valores["Major_Axis_Length"], valores["Minor_Axis_Length"]))
    if errores:
        raise ErrorValidacion(errores)
    return valores


def a_dataframe(valores: dict) -> pd.DataFrame:
    """Una fila con las columnas en el MISMO orden del entrenamiento."""
    return pd.DataFrame([[valores[c] for c in COLUMNAS]], columns=COLUMNAS)
