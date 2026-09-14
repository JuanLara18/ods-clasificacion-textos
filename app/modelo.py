# -*- coding: utf-8 -*-
"""El modelo que sirve la aplicación: lo define, lo entrena y lo guarda en disco.

    python app/modelo.py        # entrena sobre el corpus completo y lo deja en la caché

Entrenar toma unos minutos y el archivo resultante pesa unos 37 MB, así que no se versiona ni
se guarda en el Drive: queda en la caché local, junto a la de las demás herramientas del
bimestre. La aplicación lo carga en menos de un segundo, y si no lo encuentra lo entrena ella
misma la primera vez.

El pipeline está definido aquí y otra vez en el notebook, a propósito. El notebook es el
entregable y tiene que ser autocontenido, sin importar nada de este repositorio, porque el
calificador recibe dos archivos y no la carpeta. Los valores son los que cerró la búsqueda de
hiperparámetros: 500 componentes y C = 3.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "data" / "Train_textosODS.xlsx"
SEMILLA = 42

# El modelo entrenado pesa 37 MB y se regenera en un par de minutos, así que vive FUERA del
# Drive, junto a la caché de las demás herramientas del bimestre. Meterlo en el repositorio
# obligaría a sincronizar 37 MB cada vez que se reentrena.
CACHE = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "maia-tools" / "ods"
MODELO = CACHE / "modelo.joblib"

# Solo letras, dos caracteres o más: descarta números y signos.
TOKEN_LETRAS = r"(?u)\b[^\W\d_]{2,}\b"

# Palabras vacías del español, sin tildes, porque el filtro se aplica después de strip_accents.
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

# Los nombres oficiales de la Agenda 2030. El ODS 17 no está: no aparece en el corpus, así
# que el modelo no lo puede predecir nunca y conviene que eso se vea.
NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
}


def construir() -> Pipeline:
    """El mismo pipeline del notebook, con los hiperparámetros ya cerrados."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=PALABRAS_VACIAS_ES,
            token_pattern=TOKEN_LETRAS,
            ngram_range=(1, 1),
            min_df=5,
            sublinear_tf=True,
        )),
        ("svd", TruncatedSVD(n_components=500, random_state=SEMILLA)),
        ("norm", Normalizer()),
        ("clf", LogisticRegression(C=3, max_iter=2000, random_state=SEMILLA)),
    ])


def entrenar(guardar: bool = True) -> Pipeline:
    """Entrena sobre el corpus completo. La aplicación sirve, no evalúa.

    El notebook aparta un conjunto de prueba porque allí hay que medir. Aquí ya está medido,
    así que se aprovechan los 9.656 textos.
    """
    if not CORPUS.exists():
        raise SystemExit(f"no encuentro el corpus en {CORPUS}")
    corpus = pd.read_excel(CORPUS)
    modelo = construir()
    modelo.fit(corpus["textos"].astype(str), corpus["ODS"])
    if guardar:
        MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, MODELO, compress=3)
    return modelo


def cargar() -> Pipeline:
    """Devuelve el modelo guardado, y lo entrena si todavía no existe."""
    if MODELO.exists():
        return joblib.load(MODELO)
    return entrenar()


def main() -> int:
    inicio = time.time()
    print(f"entrenando sobre {CORPUS.name}, toma un par de minutos")
    modelo = entrenar()
    print(f"listo en {time.time() - inicio:.0f}s")
    print(f"guardado en {MODELO}  ({MODELO.stat().st_size / 1e6:.0f} MB)")
    ejemplo = "Las mujeres rurales enfrentan barreras para acceder a créditos agrícolas."
    print(f'prueba: "{ejemplo}" -> ODS {modelo.predict([ejemplo])[0]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
