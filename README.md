# Del dato al asistente — Clasificación de arroz Cammeo / Osmancik

Proyecto de evaluación parcial · **Tecnologías Emergentes (ISO46B)** · UNCP, Facultad de Ingeniería de Sistemas.
Grupo: **02** — 
integrantes: 
DIAZ VEGA FABRIZIO E SAUL.
TARMA CHANCO VANESSA
TOMAS ALIAGA SEBASTIA
GONZALES QUISPE JOCABED 

Sistema de punta a punta: datos → modelos (línea base, regresión logística y dos redes neuronales) → evaluación y versionado → **API Flask** → **chatbot** que la consume → pruebas con pytest.

**Dataset:** Rice (Cammeo and Osmancik), UCI ID 545, DOI 10.24432/C5MW4Z — Cinar & Koklu (2019), *Classification of Rice Varieties Using Artificial Intelligence Methods*, IJISAE 7(3):188-194. 3 810 granos, 7 medidas morfológicas. Ver `data/Citation_Request.txt`.

## Requisitos
Python 3.10 o superior. No se necesita GPU ni internet (el `.arff` ya está en `data/`).

## Instalación y ejecución desde cero
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python entrenar.py                 # 1) entrena, elige, promueve, evalúa en prueba y crea figuras (~10 s)
python -m src.api                  # 2) API + interfaz web  ->  abre http://127.0.0.1:5000 en el navegador
python -m src.chatbot              # 3) (opcional) el mismo asistente en la terminal, en OTRA terminal
pytest -q                          # 4) pruebas (datos, modelos, API, chatbot e interfaz)
```

## Interfaz web
Al levantar la API, abre **http://127.0.0.1:5000**: no hay que instalar nada más.
- **Asistente (chat):** burbujas, indicador de "escribiendo", botones de respuesta rápida (Sí / No / Ayuda / Reiniciar / Hablar con una persona), barra de progreso de las 6 medidas y una tarjeta de resultado con medidor de probabilidad. Usa el mismo chatbot de `src/chatbot.py` (mismos fallbacks y derivación).
- **Formulario rápido:** deslizadores + campos numéricos para las 6 medidas, botones de **ejemplo Cammeo / Osmancik**, dibujo del grano según largo/ancho y resaltado del campo que la API rechaza.
- Se adapta a claro/oscuro y a celular. Las conversaciones viven solo en memoria del servidor (máx. 300, se olvidan tras 30 min sin uso); no se escribe nada en disco.
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
  preprocesamiento_api.py  validación de entradas de la API
  api.py                   servicio Flask (/salud, /info, /predecir)
  chatbot.py               asistente conversacional por reglas (slots, fallback, derivación)
  web/index.html           interfaz web (chat + formulario), servida por la API en /
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
