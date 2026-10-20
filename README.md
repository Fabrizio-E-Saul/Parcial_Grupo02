# Del dato al asistente — Clasificación de arroz Cammeo / Osmancik

Proyecto de evaluación parcial · **Tecnologías Emergentes (ISO46B)** · UNCP, Facultad de Ingeniería de Sistemas.
Grupo: **XX** — integrantes: _(completar)_.

Sistema de punta a punta: datos → modelos (línea base, regresión logística y dos redes neuronales) → evaluación y versionado → **API Flask** → **chatbot** que la consume → pruebas con pytest.

**Dataset:** Rice (Cammeo and Osmancik), UCI ID 545, DOI 10.24432/C5MW4Z — Cinar & Koklu (2019), *Classification of Rice Varieties Using Artificial Intelligence Methods*, IJISAE 7(3):188-194. 3 810 granos, 7 medidas morfológicas. Ver `data/Citation_Request.txt`.

## Requisitos
Python 3.10 o superior. No se necesita GPU ni internet (el `.arff` ya está en `data/`).

## Instalación y ejecución desde cero
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python entrenar.py                 # 1) entrena, elige, promueve, evalúa en prueba y crea figuras (~10 s)
python -m src.api                  # 2) API + interfaz web  ->  abre http://127.0.0.1:5000 en el navegador
python -m src.chatbot              # 3) (opcional) el mismo asistente en la terminal, en OTRA terminal
pytest -q                          # 4) pruebas (datos, modelos, API, chatbot e interfaz)
```

## Interfaz web
Al levantar la API, abre **http://127.0.0.1:5000**: no hay que instalar nada más.
- **Asistente (chat):** burbujas, indicador de "escribiendo", botones de respuesta rápida (Sí / No / Ayuda / Reiniciar / Hablar con una persona), barra de progreso de las 6 medidas y tarjeta de resultado con medidor de probabilidad. Usa el mismo chatbot de `src/chatbot.py`. Mensajes cortos; a la derecha hay una **reseña** que saluda y explica el propósito del asistente.
- **Teclado restringido:** cuando el bot pide una medida solo se pueden escribir números (un punto o coma decimal); en el resto de la conversación, solo letras. Lo pegado también se filtra.
- **Límites por el dataset:** cada medida debe estar entre el **mínimo y el máximo del dataset completo** (se muestran en la pregunta y bajo cada campo). Además, el **área convexa no puede ser menor que el área**, el **ancho no puede ser mayor que el largo** y el **área y el perímetro deben ser coherentes con el largo y el ancho** (un grano real es casi una elipse; si no cuadran, se explica qué valor se esperaba). Lo validan la interfaz, el chatbot y la API (400 con mensaje claro).
- **Reiniciar / Borrar historial:** *Reiniciar* empieza una consulta nueva y deja el historial a la vista; *Borrar historial* limpia la pantalla y elimina la sesión del servidor.
- **Hablar con una persona:** muestra botones de **WhatsApp** de los integrantes (se configuran en `src/config.py`).
- **Formulario rápido:** deslizadores + campos numéricos que respetan las mismas reglas, botones de **ejemplo Cammeo / Osmancik**, dibujo del grano y resaltado del campo que la API rechaza.
- Se adapta a claro/oscuro y a celular. Las conversaciones viven solo en memoria del servidor (máx. 300, se olvidan tras 30 min sin uso); no se escribe nada en disco.

### Dónde personalizar
| Qué | Dónde |
|---|---|
| WhatsApp de los integrantes y mensaje inicial | `src/config.py` → `INTEGRANTES`, `MENSAJE_WHATSAPP` |
| Frases del chatbot (saludo, resultado, errores...) y las 6 preguntas | `src/config.py` → `TEXTOS`, `SLOTS` |
| Reseña al costado (saludo y propósito) y título | `src/web/index.html` → bloque `<aside class="panel acerca">` y `<header>` |
| Colores | `src/web/index.html` → bloque `🎨 COLORES Y FONDO` al inicio de `<style>` |
| Logo FIS (arriba izquierda) y logo UNCP (arriba derecha) | guardar `src/web/img/logo_fis.jpg` y `src/web/img/logo_uncp.png` (encabezado de `index.html`) |
| Foto de cada arroz en el resultado | guardar `src/web/img/arroz_cammeo.jpg` y `src/web/img/arroz_osmancik.jpg` (prueba también .png y .webp; sin foto se dibuja un grano) |
| Diagrama de flujo del chatbot | editar `docs/generar_diagrama.py` y ejecutarlo: regenera `docs/diagrama_flujo_chatbot.png` |
| Foto del grupo | guardar `src/web/img/grupo.jpg` (sale en la reseña); para usarla de fondo, poner `--foto-grupo: url("/img/grupo.jpg");` |

`entrenar.py` **borra y recrea `modelos/` en cada corrida** y evalúa la prueba una sola vez por corrida. Con la misma semilla (42) los resultados son idénticos.

### Probar la API sin el chatbot
```bash
curl http://127.0.0.1:5000/salud
curl -X POST http://127.0.0.1:5000/predecir -H "Content-Type: application/json" \
  -d '{"Area":14162,"Perimeter":487,"Major_Axis_Length":205,"Minor_Axis_Length":88,"Convex_Area":14494,"Extent":0.65}'
```
Contrato completo en `GET /info`. Errores de entrada → **400** con mensajes en español (`{"error":..., "detalles":[...]}`).

## Estructura
```
README.md · requirements.txt · entrenar.py
src/
  datos.py                 carga .arff, limpieza, partición 60/20/20 estratificada
  modelo.py                Pipeline(StandardScaler -> modelo): trivial, logística, mlp_v1, mlp_v2
  evaluacion.py            métricas, umbral por costo, variabilidad, fichas JSON, promoción, prueba única
  preprocesamiento_api.py  validación de entradas de la API (rango del dataset, relaciones entre medidas)
  api.py                   servicio Flask (/salud, /info, /predecir, /chat, /contactos)
  config.py                contactos WhatsApp, textos del chatbot y medidas (editable)
  chatbot.py               asistente conversacional por reglas (slots, fallback, derivación)
  web/index.html           interfaz web (chat + reseña + formulario), servida por la API en /
  web/img/                 foto del grupo (grupo.jpg)
  figuras.py               figuras del informe
tests/                     pytest: datos, modelo/evaluación, API, chatbot, interfaz
modelos/                   .joblib + ficha .json por versión, promovida.json, prueba_evaluada.json
docs/                      diagrama_flujo_chatbot.png, figuras/, latencia_api.json
notebooks/                 01_exploracion_y_modelado.ipynb (ejecutado)
data/                      Rice_Cammeo_Osmancik.arff, Citation_Request.txt
informe.pdf
```

## Decisiones clave (resumen; detalle en el informe)
- **Clase positiva = Cammeo.** Supuesto de costos del escenario: falso negativo (Cammeo → Osmancik) = 2, falso positivo = 1. *No son precios reales.* El umbral se elige **en validación**.
- **Sin fuga:** el escalador vive dentro del `Pipeline` y solo ve entrenamiento; se comprueba que no hay filas repetidas entre particiones.
- **Elección de la versión** solo con validación; regla fijada de antemano (menor costo; empate de ±0.01 → el más simple). **La prueba se mira una vez** y el código impide una segunda.
- **Redes neuronales:** `MLPClassifier` de scikit-learn (L2 + parada temprana). No se usa dropout ni pesos de clase porque esa clase no los ofrece.
- **Eccentricity** es función exacta de los ejes: la API la calcula si no se envía.
- **Chatbot:** no guarda datos; deriva a una persona tras 3 respuestas ilegibles, 3 rechazos de la API o si la API no responde.

## Reproducibilidad
Semilla fija (`SEMILLA = 42` en `src/datos.py`). Cada ficha JSON guarda: fecha, hash SHA-256 del archivo de datos, parámetros, umbral, métricas, variabilidad y versiones de librerías.

## Declaración de uso de herramientas de IA
Ver la sección *Anexo A* de `informe.pdf`.
