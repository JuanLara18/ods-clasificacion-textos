# -*- coding: utf-8 -*-
"""Aplicación interactiva: un texto en español entra, un ODS sale.

    streamlit run app/app.py

Usa el mismo pipeline del notebook, definido en modelo.py. El segundo objetivo se propone solo
cuando es plausible, y los dos se muestran juntos cuando quedan empatados: muchos textos tocan
varios objetivos, y eso es lo que el proyecto encontró que la etiqueta única esconde.
"""

import sys
from pathlib import Path

# El repositorio vive en el Drive y ahí no se escriben .pyc. Hay que fijarlo antes de importar
# modelo.py, que es lo único nuestro que se importa.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st

from modelo import NOMBRES_ODS, cargar

EJEMPLO = (
    "El plan de saneamiento básico prioriza la construcción de redes de alcantarillado y "
    "plantas de tratamiento de aguas residuales en los municipios que hoy vierten directamente "
    "a los ríos. Se busca ampliar la cobertura de agua potable y mejorar la gestión de los "
    "recursos hídricos mediante la medición del consumo y la reducción de pérdidas en la red "
    "de distribución."
)

# Bajo este margen entre el primero y el segundo, el texto se declara transversal.
MARGEN_ESTRECHO = 0.10
# Bajo esta probabilidad el segundo objetivo no aporta nada, y mostrarlo resta credibilidad.
UMBRAL_SEGUNDO = 0.10
PALABRAS_MINIMAS = 20


def porcentaje(valor: float) -> str:
    """Con coma decimal, como se escriben los números en español."""
    return f"{valor:.1%}".replace(".", ",")


@st.cache_resource(show_spinner="Preparando el modelo. La primera vez lo entrena, y eso toma "
                                "un par de minutos.")
def obtener_modelo():
    return cargar()


def poner_ejemplo() -> None:
    st.session_state.texto = EJEMPLO


def barra_lateral() -> None:
    with st.sidebar:
        st.subheader("El modelo")
        st.write("TF-IDF, SVD truncada a 1.000 componentes y regresión logística.")

        # Las tres en fila no caben: la barra lateral tiene ancho fijo y st.metric recorta
        # el valor a "0,...". Van una debajo de otra, que es lo que sí se lee.
        for etiqueta, valor in [("Exactitud", "0,885"), ("F1 macro", "0,859"),
                                ("Top 2", "0,959")]:
            st.metric(etiqueta, valor)
        st.caption("Medido sobre 1.932 textos no vistos. El modelo que responde aquí se "
                   "entrenó después con el corpus completo.")

        st.divider()
        st.caption("Son 16 objetivos y no 17: el ODS 17 no aparece en el corpus, así que el "
                   "sistema nunca lo propone.")


def mostrar_resultado(texto: str, modelo) -> None:
    probabilidades = modelo.predict_proba([texto])[0]
    orden = np.argsort(-probabilidades)
    clases = modelo.classes_

    primero, segundo = clases[orden[0]], clases[orden[1]]
    p_primero, p_segundo = probabilidades[orden[0]], probabilidades[orden[1]]

    with st.container(border=True):
        objetivo, probabilidad = st.columns([3, 1], vertical_alignment="center")
        objetivo.caption(f"ODS {primero}")
        objetivo.markdown(f"### {NOMBRES_ODS[primero]}")
        probabilidad.metric("Probabilidad", porcentaje(p_primero))

    if p_primero - p_segundo < MARGEN_ESTRECHO:
        st.info(
            f"**Texto transversal.** El ODS {primero} y el ODS {segundo}, "
            f"{NOMBRES_ODS[segundo].lower()}, quedan casi empatados "
            f"({porcentaje(p_primero)} y {porcentaje(p_segundo)}). Conviene considerar los dos."
        )
    elif p_segundo >= UMBRAL_SEGUNDO:
        st.caption(
            f"También podría ser ODS {segundo}, {NOMBRES_ODS[segundo].lower()} "
            f"({porcentaje(p_segundo)})."
        )

    with st.expander("Los cinco más probables"):
        for posicion in orden[:5]:
            ods = clases[posicion]
            st.progress(
                float(probabilidades[posicion]),
                text=f"ODS {ods}, {NOMBRES_ODS[ods]} ({porcentaje(probabilidades[posicion])})",
            )


def main() -> None:
    st.set_page_config(page_title="Clasificador de textos ODS", layout="centered")
    barra_lateral()

    st.title("Clasificador de textos ODS")
    st.caption("Relaciona un párrafo en español con uno de los 16 Objetivos de Desarrollo "
               "Sostenible de la Agenda 2030.")

    st.text_area(
        "Texto",
        key="texto",
        height=180,
        label_visibility="collapsed",
        placeholder="Pegue aquí un párrafo: un fragmento de un plan de desarrollo, un informe "
                    "de política pública o una propuesta ciudadana.",
    )

    clasificar, ejemplo, _ = st.columns([1, 1, 2])
    pulsado = clasificar.button("Clasificar", type="primary", width="stretch")
    ejemplo.button("Ejemplo", on_click=poner_ejemplo, width="stretch")

    if not pulsado:
        return

    texto = st.session_state.texto.strip()
    if not texto:
        st.warning("Escriba un texto para clasificarlo.")
        return

    if len(texto.split()) < PALABRAS_MINIMAS:
        st.warning(
            f"Con menos de {PALABRAS_MINIMAS} palabras la predicción es poco confiable: el "
            "modelo aprendió de párrafos de unas 105."
        )

    modelo = obtener_modelo()

    # Sin ningún término del vocabulario el TF-IDF queda vacío y el modelo respondería solo con
    # sus sesgos: se avisa en vez de clasificar.
    if modelo.named_steps["tfidf"].transform([texto]).nnz == 0:
        st.error("El texto no contiene ninguna palabra del vocabulario del modelo, así que no hay "
                 "base para clasificarlo. El modelo trabaja con párrafos en español.")
        return

    mostrar_resultado(texto, modelo)


main()
