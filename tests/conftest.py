import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import datos as D  
from src.api import crear_app  


@pytest.fixture(scope="session")
def particiones():
    p, calidad = D.cargar_particiones()
    return p, calidad


@pytest.fixture(scope="session")
def directorio_modelos():
    d = RAIZ / "modelos"
    if not (d / "promovida.json").exists():
        import entrenar
        entrenar.main(con_figuras=False)
    return d


@pytest.fixture(scope="session")
def cliente_api(directorio_modelos):
    app = crear_app(directorio_modelos)
    app.testing = True
    return app.test_client()


GRANO_CAMMEO = {"Area": 14162, "Perimeter": 487, "Major_Axis_Length": 205,
                "Minor_Axis_Length": 88, "Convex_Area": 14494, "Extent": 0.65}


@pytest.fixture()
def grano():
    return dict(GRANO_CAMMEO)
