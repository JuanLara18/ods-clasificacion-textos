"""Perfila el corpus de textos ODS antes de tocar el modelo.

    python scripts/perfilar_corpus.py

Responde lo que hay que saber antes de decidir la preparación: cuántos textos hay, qué
clases aparecen de verdad, qué tan desbalanceadas están, de qué largo son los documentos y
si la aumentación con ChatGPT dejó textos casi repetidos. Ese último punto es el que puede
inflar el desempeño si dos versiones del mismo texto caen a lado y lado de la partición.

No transforma nada ni escribe en data/: solo mide e imprime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "data" / "Train_textosODS.xlsx"

# El enunciado habla de 17 ODS; el archivo entregado solo trae del 1 al 16.
ODS_ESPERADOS = set(range(1, 18))


def cargar() -> pd.DataFrame:
    df = pd.read_excel(CORPUS)
    faltan = {"textos", "ODS"} - set(df.columns)
    if faltan:
        raise SystemExit(f"al archivo le faltan columnas: {sorted(faltan)}")
    return df


def describir_clases(df: pd.DataFrame) -> None:
    conteo = df["ODS"].value_counts().sort_values(ascending=False)
    presentes = set(int(o) for o in conteo.index)
    ausentes = sorted(ODS_ESPERADOS - presentes)

    print(f"\ntextos: {len(df):,}".replace(",", "."))
    print(f"clases presentes: {len(presentes)} de 17")
    if ausentes:
        print(f"ODS que no aparecen: {ausentes}")

    razon = conteo.max() / conteo.min()
    print(f"desbalance mayor/menor: {razon:.1f} a 1 "
          f"(ODS {conteo.idxmax()} con {conteo.max()}, ODS {conteo.idxmin()} con {conteo.min()})")
    print("\ndistribución por clase")
    for ods, n in conteo.items():
        print(f"  ODS {int(ods):2d}  {n:5d}  {n / len(df):6.1%}  {'#' * round(n / 25)}")


def describir_textos(df: pd.DataFrame) -> None:
    palabras = df["textos"].astype(str).str.split().str.len()
    caracteres = df["textos"].astype(str).str.len()
    print("\nlongitud de los documentos")
    print(f"  palabras    min {palabras.min():3d}   p25 {palabras.quantile(.25):.0f}   "
          f"mediana {palabras.median():.0f}   p75 {palabras.quantile(.75):.0f}   max {palabras.max()}")
    print(f"  caracteres  min {caracteres.min():3d}   mediana {caracteres.median():.0f}   "
          f"max {caracteres.max()}")

    nulos = df.isna().sum()
    print(f"\nnulos: textos {nulos['textos']}, ODS {nulos['ODS']}")
    print(f"textos exactamente repetidos: {df.duplicated(subset=['textos']).sum()}")


def buscar_casi_duplicados(df: pd.DataFrame, umbral: float = 0.9, muestra: int = 2000) -> None:
    """Cuenta pares muy parecidos dentro de una misma clase, sobre una muestra.

    La aumentación con la API de ChatGPT genera variantes de un mismo texto. Si dos
    variantes quedan una en entrenamiento y otra en prueba, la evaluación mide memoria y
    no generalización. Aquí solo se cuantifica el riesgo; la decisión de qué hacer va al
    notebook.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sub = df.sample(min(muestra, len(df)), random_state=0)
    X = TfidfVectorizer(min_df=2).fit_transform(sub["textos"].astype(str))

    total = 0
    for ods, idx in sub.reset_index(drop=True).groupby("ODS").groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        S = cosine_similarity(X[idx])
        for i in range(len(idx)):
            S[i, i] = 0.0
        total += int((S >= umbral).sum() // 2)

    print(f"\npares con coseno TF-IDF >= {umbral} dentro de la misma clase, "
          f"sobre una muestra de {len(sub)}: {total}")
    if total:
        print("  hay variantes casi idénticas: partir con cuidado, ver Enunciado.md")


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"no está el corpus en {CORPUS}, se descarga de Coursera")
    df = cargar()
    describir_clases(df)
    describir_textos(df)
    buscar_casi_duplicados(df)


if __name__ == "__main__":
    main()
