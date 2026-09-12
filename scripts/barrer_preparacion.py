# -*- coding: utf-8 -*-
"""Mide, una decisión a la vez, cómo afecta la preparación del texto al desempeño.

    python scripts/barrer_preparacion.py
    python scripts/barrer_preparacion.py --rapido    # solo los factores, sin la rejilla

El criterio del 30% del enunciado pide justificar las decisiones de preparación. Este script
las mide en vez de suponerlas: parte de una configuración base y cambia un factor cada vez,
con validación cruzada estratificada de cinco particiones sobre la partición de entrenamiento
y una regresión logística fija como instrumento de medida.

El conjunto de prueba no se toca: se separa con la misma semilla y el mismo test_size del
notebook, y aquí solo se usa la parte de entrenamiento.

La lista de palabras vacías está duplicada en el notebook a propósito. El notebook tiene que
ser autocontenido y no puede importar nada de este repositorio, porque el calificador recibe
dos archivos y no la carpeta.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "data" / "Train_textosODS.xlsx"
SEMILLA = 42

# Solo letras, dos caracteres o más: descarta números y signos. La clase [^\W\d_] es
# "carácter de palabra que no es dígito ni guion bajo", es decir letra, con tildes incluidas.
TOKEN_LETRAS = r"(?u)\b[^\W\d_]{2,}\b"
TOKEN_CON_NUMEROS = r"(?u)\b\w\w+\b"  # el de scikit-learn por omisión

# Palabras vacías del español, sin tildes. TfidfVectorizer no trae lista para español y el
# filtro se aplica DESPUÉS de quitar las tildes, así que la lista tiene que venir sin ellas
# o no empareja nada.
PALABRAS_VACIAS_ES = [
    "a", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algun", "ambos", "ante",
    "antes", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "asi", "aun",
    "aunque", "bajo", "bien", "cada", "casi", "como", "con", "contra", "cual", "cuales",
    "cualquier", "cuando", "cuanto", "cuya", "cuyo", "de", "debe", "deben", "del", "demas",
    "dentro", "desde", "despues", "donde", "dos", "durante", "e", "el", "ella", "ellas",
    "ello", "ellos", "en", "entre", "era", "eran", "eres", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estaba", "estaban", "estado", "estan", "estar", "estas", "este", "esto",
    "estos", "estoy", "fue", "fueron", "fui", "ha", "habia", "habian", "han", "hace", "hacen",
    "hacer", "hacia", "hasta", "hay", "he", "incluso", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mediante", "menos", "mi", "mientras", "mis", "mucha", "muchas",
    "mucho", "muchos", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestras",
    "nuestro", "nuestros", "o", "otra", "otras", "otro", "otros", "para", "pero", "poco",
    "pocos", "por", "porque", "puede", "pueden", "pues", "que", "quien", "quienes", "se",
    "sea", "sean", "segun", "ser", "si", "sido", "siendo", "sin", "sino", "sobre", "sola",
    "solamente", "solo", "son", "su", "sus", "tal", "tambien", "tampoco", "tan", "tanto",
    "te", "tiene", "tienen", "tener", "toda", "todas", "todo", "todos", "tras", "tu", "tus",
    "un", "una", "unas", "uno", "unos", "usted", "ustedes", "ya", "yo",
]

BASE = dict(
    lowercase=True,
    strip_accents="unicode",
    stop_words=PALABRAS_VACIAS_ES,
    token_pattern=TOKEN_LETRAS,
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.9,
    sublinear_tf=False,
)

# Un factor a la vez desde BASE. La etiqueta dice qué decisión se está midiendo.
FACTORES = [
    ("base", {}),
    ("sin quitar palabras vacias", dict(stop_words=None)),
    ("conservando las tildes", dict(strip_accents=None)),
    ("conservando los numeros", dict(token_pattern=TOKEN_CON_NUMEROS)),
    ("con bigramas", dict(ngram_range=(1, 2))),
    ("min_df = 1", dict(min_df=1)),
    ("min_df = 3", dict(min_df=3)),
    ("min_df = 5", dict(min_df=5)),
    ("max_df = 1.0", dict(max_df=1.0)),
    ("max_df = 0.5", dict(max_df=0.5)),
    ("sublinear_tf", dict(sublinear_tf=True)),
]

# Combinaciones de lo que haya salido bien por separado, para ver si se suma o se estorba.
REJILLA = [
    # La que se llevó al notebook, cerrada el 12 de septiembre de 2026.
    ("min_df = 5 + sublinear_tf", dict(min_df=5, sublinear_tf=True)),
    ("bigramas + sublinear_tf", dict(ngram_range=(1, 2), sublinear_tf=True)),
    ("bigramas + min_df = 3", dict(ngram_range=(1, 2), min_df=3)),
    ("bigramas + sublinear_tf + min_df = 3",
     dict(ngram_range=(1, 2), sublinear_tf=True, min_df=3)),
    ("sublinear_tf + numeros", dict(sublinear_tf=True, token_pattern=TOKEN_CON_NUMEROS)),
    ("sublinear_tf + min_df = 1", dict(sublinear_tf=True, min_df=1)),
]


def leer_argumentos():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rapido", action="store_true",
                        help="omite la rejilla de combinaciones")
    return parser.parse_args()


def medir(X, y, **cambios) -> tuple[float, float, int, float]:
    """Devuelve F1 macro, exactitud, tamaño del vocabulario y segundos."""
    parametros = {**BASE, **cambios}
    tubo = Pipeline([
        ("tfidf", TfidfVectorizer(**parametros)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
    ])
    inicio = time.time()
    puntajes = cross_validate(
        tubo, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA),
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )
    segundos = time.time() - inicio
    vocabulario = len(TfidfVectorizer(**parametros).fit(X).vocabulary_)
    return (puntajes["test_f1_macro"].mean(), puntajes["test_accuracy"].mean(),
            vocabulario, segundos)


def imprimir_tabla(titulo: str, filas: list, referencia: float) -> None:
    print(f"\n{titulo}")
    print(f"  {'configuracion':<38} {'F1 macro':>9} {'exactitud':>10} "
          f"{'vocab':>8} {'vs base':>9}")
    for etiqueta, f1, exactitud, vocabulario, _ in filas:
        print(f"  {etiqueta:<38} {f1:>9.4f} {exactitud:>10.4f} "
              f"{vocabulario:>8d} {f1 - referencia:>+9.4f}")


def main() -> int:
    argumentos = leer_argumentos()
    corpus = pd.read_excel(CORPUS)
    X_train, _, y_train, _ = train_test_split(
        corpus["textos"].astype(str), corpus["ODS"],
        test_size=0.2, stratify=corpus["ODS"], random_state=SEMILLA,
    )
    print(f"{len(X_train)} textos de entrenamiento, {y_train.nunique()} clases")
    print("instrumento de medida: regresion logistica, validacion cruzada estratificada de 5")
    print()

    filas = []
    for etiqueta, cambios in FACTORES:
        f1, exactitud, vocabulario, segundos = medir(X_train, y_train, **cambios)
        filas.append((etiqueta, f1, exactitud, vocabulario, segundos))
        print(f"  {etiqueta:<38} F1 {f1:.4f}  ({segundos:.0f}s)")
    referencia = filas[0][1]
    imprimir_tabla("un factor a la vez, desde la configuracion base", filas, referencia)

    if not argumentos.rapido:
        print()
        combinadas = []
        for etiqueta, cambios in REJILLA:
            f1, exactitud, vocabulario, segundos = medir(X_train, y_train, **cambios)
            combinadas.append((etiqueta, f1, exactitud, vocabulario, segundos))
            print(f"  {etiqueta:<38} F1 {f1:.4f}  ({segundos:.0f}s)")
        imprimir_tabla("combinaciones", combinadas, referencia)
        filas = filas + combinadas

    mejor = max(filas, key=lambda fila: fila[1])
    print(f"\nmejor: {mejor[0]}  F1 macro {mejor[1]:.4f}  exactitud {mejor[2]:.4f}")
    print("La configuracion que se lleve al notebook se escribe en docs/decisiones.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
