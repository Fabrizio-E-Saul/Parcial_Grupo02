
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

VERDE, AMBAR, ROJO, GRIS = "#d5f0dc", "#fdebc8", "#f8d3d0", "#e6e9ee"
BORDE = {VERDE: "#2e8b57", AMBAR: "#d68910", ROJO: "#c0392b", GRIS: "#566573"}
TXT_ROJO = "#a93226"

fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 170); ax.set_ylim(-3, 100); ax.axis("off")


def caja(x, y, w, h, texto, color, fs=8.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=color, ec=BORDE[color], lw=1.4))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs)


def rombo(x, y, w, h, texto, color=AMBAR, fs=8):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         fc=color, ec=BORDE[color], lw=1.4))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs)


def conector(x, y, letra="A", color=AMBAR):
    ax.add_patch(Circle((x, y), 2.4, fc="white", ec=BORDE[color], lw=1.6))
    ax.text(x, y, letra, ha="center", va="center", fontsize=9, fontweight="bold")


def flecha(p, q, texto="", color="#34495e", dx=0, dy=0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    if texto:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, texto, fontsize=7.8, color=color,
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))


ax.text(85, 97, "Diagrama de flujo del chatbot de acopio de arroz", fontsize=13, fontweight="bold", ha="center")

caja(9, 86, 14, 8, "INICIO\nusuario escribe\n'hola'", GRIS)
caja(33, 86, 22, 11, "Saludo + privacidad\n'No guardo tus datos'\n¿Empezamos? (sí / no)", VERDE)
rombo(58, 86, 14, 10, "¿sí?")
caja(86, 86, 22, 11, "Pide una medida (n/6)\ncon explicación, ejemplo\ny rango del dataset", VERDE)
rombo(116, 86, 26, 15, "¿valor válido?\n(número, rango y\ncoherencia: ver\ncaja de reglas)", fs=7.6)
caja(146, 86, 20, 9, "Guarda el dato\ny pasa a la\nsiguiente", VERDE)
flecha((16, 86), (22, 86)); flecha((44, 86), (51, 86)); flecha((65, 86), (75, 86), "sí", dy=2.2)
flecha((97, 86), (103, 86)); flecha((129, 86), (136, 86), "sí", dy=2.2)
conector(86, 68, "A")
flecha((86, 70.4), (86, 81), color="#d68910"); ax.text(89, 75, "destino del\nconector A", fontsize=7, color="#7e5109", va="center")

flecha((58, 81), (58, 70.5), "no", dx=2.5)
caja(58, 66, 20, 7, "Despedida y FIN", VERDE)

flecha((116, 78.5), (116, 72.5), "no", color=TXT_ROJO, dx=2.5)
rombo(116, 66, 22, 12, "fallos + 1\n¿llegó a 3?", ROJO)
flecha((116, 60), (116, 52), "sí", color=TXT_ROJO, dx=2.5)
caja(116, 48, 24, 9, "DERIVA A UNA PERSONA\n(botones de WhatsApp\ndel equipo)", ROJO, 7.8)
flecha((105, 66), (100.5, 66), "no: repite la\npregunta\n(quedan N)", color=TXT_ROJO, dy=6.5)
conector(98, 66, "A", ROJO)

flecha((146, 81.5), (146, 72.5))
rombo(146, 66, 20, 12, "¿faltan\nmedidas?")
flecha((156, 66), (160.5, 66), "sí", dy=2.2)
conector(163, 66, "A")
flecha((146, 60), (146, 53.5), "no", dx=2.5)
caja(146, 49, 20, 8, "POST /predecir\na la API", VERDE)
flecha((146, 45), (146, 37.5))
rombo(146, 31, 22, 13, "Respuesta\nde la API")

flecha((135, 31), (127, 31), "200", dy=2.2)
caja(112, 31, 28, 14, "Resultado + imagen del arroz\n(variedad + seguridad %)\nSi la confianza es baja: recomienda\nmedir otra vez / revisión humana\nRecuerda sus límites · borra medidas", VERDE, 7.8)
flecha((98, 31), (85, 31))
rombo(76, 31, 16, 11, "¿otro\ngrano?")
flecha((76, 36.5), (76, 41.5), "sí", dx=2.5)
caja(76, 45, 24, 6, "Reinicia contadores", AMBAR, 7.8)
flecha((76, 48.2), (76, 49.8)); conector(76, 52.2, "A")
flecha((76, 25.5), (76, 18), "no", dx=2.5)
caja(76, 14, 22, 7, "Despedida y FIN", VERDE)

flecha((146, 24.5), (146, 19.5), "400 / sin respuesta", color=TXT_ROJO, dx=-13, dy=0)
caja(146, 12, 28, 13, "Error de la API\n400: dice qué dato falla, lo borra\ny lo vuelve a pedir (máx. 3)\nAPI caída: avisa con claridad", ROJO, 7.8)
flecha((132, 12), (124, 12), "3 rechazos\no API caída", color=TXT_ROJO, dx=-3, dy=6.5)
caja(113, 12, 22, 9, "DERIVA A UNA PERSONA\n(botones de WhatsApp)", ROJO, 7.8)
flecha((160, 12), (164.6, 12), "corrige", dy=2.4)
conector(167, 12, "A")

caja(22, 62, 34, 22, "EN CUALQUIER MOMENTO\n'ayuda' → repite la pregunta\n'reiniciar' → borra todo y vuelve a 1/6\n'cancelar' → FIN sin guardar nada\n'persona' → deriva (WhatsApp)\nWeb: botón Reiniciar (conserva historial)\nWeb: botón Borrar historial (limpia todo)", AMBAR, 7.6)
caja(22, 40, 32, 11, "DATOS PERSONALES\n(correo, teléfono): se ignoran\ny se avisa; nunca se guardan", AMBAR, 7.8)

caja(22, 18, 36, 20, "UNA MEDIDA SE RECHAZA SI:\n· no es un número, o es ≤ 0\n· está fuera del mín–máx del dataset\n· ancho > largo, o convexa < área\n· área o perímetro no cuadran con\n  largo y ancho (grano imposible):\n  borra área, perímetro y largo\n→ fallos + 1 (3 seguidos: persona)", ROJO, 7.4)

for i, (c, t) in enumerate([(VERDE, "Camino feliz"), (AMBAR, "Camino alternativo"), (ROJO, "Camino de error")]):
    ax.add_patch(FancyBboxPatch((4 + i * 24, 1.5), 4, 3, boxstyle="round,pad=0.2", fc=c, ec=BORDE[c]))
    ax.text(9.5 + i * 24, 3, t, fontsize=8.5, va="center")
ax.text(110, 3, "(A) = conector: todos los círculos A de origen continúan en el A situado bajo 'Pide una medida'",
        fontsize=7.8, ha="center", color="#444")
fig.savefig(Path(__file__).parent / "diagrama_flujo_chatbot.png", dpi=130, bbox_inches="tight")
