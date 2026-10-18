import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import datos as D


def test_carga_dimensiones_y_clases(particiones):
    df = D.cargar_crudo()
    assert df.shape == (3810, 8)
    assert set(df[D.OBJETIVO].unique()) == {"Cammeo", "Osmancik"}  


def test_reporte_de_calidad_del_dataset_real(particiones):
    _, calidad = particiones
    assert calidad["filas_iniciales"] == 3810
    assert calidad["nulos"] == 0 and calidad["duplicados"] == 0 and calidad["incoherentes"] == 0
    assert calidad["distribucion_clases"] == {"Osmancik": 2180, "Cammeo": 1630}


def _df_base():
    return pd.DataFrame({
        "Area": [100., 100., 200., 300., 400.], "Perimeter": [50.] * 5,
        "Major_Axis_Length": [20., 20., 25., 30., 10.], "Minor_Axis_Length": [10., 10., 12., 15., 20.],
        "Eccentricity": [0.5, 0.5, 0.5, 0.5, 0.5], "Convex_Area": [110., 110., 210., 250., 410.],
        "Extent": [0.6, 0.6, 0.7, 0.5, 0.5], "Class": ["Cammeo", "Cammeo", "Osmancik", "Osmancik", "Cammeo"],
    })


def test_limpiar_quita_duplicados_e_incoherentes():
    df = _df_base()
    limpio, rep = D.limpiar(df)
    assert rep["duplicados"] == 1          
    assert rep["incoherentes"] == 2        
    assert len(limpio) == 2


def test_limpiar_quita_nulos():
    df = _df_base()
    df.loc[2, "Area"] = np.nan
    _, rep = D.limpiar(df)
    assert rep["nulos"] == 1


def test_particion_tamanos_y_estratificacion(particiones):
    p, _ = particiones
    assert (len(p["X_ent"]), len(p["X_val"]), len(p["X_pru"])) == (2286, 762, 762)
    base = 1630 / 3810
    for k in ("y_ent", "y_val", "y_pru"):
        assert abs(p[k].mean() - base) < 0.005   


def test_sin_filas_repetidas_entre_particiones(particiones):
    p, _ = particiones
    assert D.comprobar_sin_fuga(p) == {"ent_val": 0, "ent_pru": 0, "val_pru": 0}


def test_particion_reproducible_con_misma_semilla():
    a, _ = D.cargar_particiones(semilla=7)
    b, _ = D.cargar_particiones(semilla=7)
    c, _ = D.cargar_particiones(semilla=8)
    assert a["X_pru"].index.equals(b["X_pru"].index)
    assert not a["X_pru"].index.equals(c["X_pru"].index)


def test_escalador_se_ajusta_solo_con_entrenamiento(particiones):
    p, _ = particiones
    esc = StandardScaler().fit(p["X_ent"])
    todo = pd.concat([p["X_ent"], p["X_val"], p["X_pru"]])
    assert np.allclose(esc.mean_, p["X_ent"].mean().values)
    assert not np.allclose(esc.mean_, todo.mean().values, atol=1e-9)  


def test_archivo_inexistente_lanza_error():
    with pytest.raises(Exception):
        D.cargar_crudo(Path(tempfile.gettempdir()) / "no_existe_12345.arff")


def test_hash_del_archivo_es_estable():
    assert D.hash_archivo() == D.hash_archivo() and len(D.hash_archivo()) == 64
