# -*- coding: utf-8 -*-
"""Aplicación interactiva: un texto en español entra, un ODS sale.

    streamlit run app/app.py

Usa el mismo pipeline del notebook, definido en modelo.py. Muestra los dos objetivos más
probables y no solo el primero, porque muchos textos tocan varios a la vez: es la conclusión
del proyecto llevada a la interfaz.
"""

import sys
from pathlib import Path

# El repositorio vive en el Drive y ahí no se escriben .pyc, según AGENTS.md. Hay que fijarlo
# antes de importar modelo.py, que es lo único nuestro que se importa.
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

MARGEN_ESTRECHO = 0.10
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
        st.write(
            "Bolsa de palabras con pesado TF-IDF sobre 8.353 términos, SVD truncada a 500 "
            "componentes, normalización de las filas y regresión logística."
        )
        izquierda, centro, derecha = st.columns(3)
        izquierda.metric("Exactitud", "0,876")
        centro.metric("F1 macro", "0,848")
        derecha.metric("Top 2", "0,948")
        st.caption(
            "Medido con este mismo pipeline entrenado sobre el 80% del corpus y evaluado "
            "sobre el 20% restante. El modelo que responde aquí se entrenó con el corpus "
            "completo, así que vio más textos que el que se midió."
        )

        st.divider()
        st.subheader("Lo que conviene saber")
        st.write(
            "**Son 16 objetivos y no 17.** El ODS 17, alianzas para lograr los objetivos, no "
            "aparece en el corpus de entrenamiento, así que el sistema no lo puede proponer."
        )
        st.write(
            "**Muchos textos tocan varios objetivos.** Por eso se muestran los dos más "
            "probables, y no solo el primero."
        )


def mostrar_resultado(texto: str, modelo) -> None:
    probabilidades = modelo.predict_proba([texto])[0]
    orden = np.argsort(-probabilidades)
    clases = modelo.classes_

    primero, segundo = clases[orden[0]], clases[orden[1]]
    p_primero, p_segundo = probabilidades[orden[0]], probabilidades[orden[1]]

    with st.container(border=True):
        objetivo, probabilidad = st.columns([3, 1])
        objetivo.markdown(f"### ODS {primero}")
        objetivo.markdown(f"**{NOMBRES_ODS[primero]}**")
        probabilidad.metric("Probabilidad", porcentaje(p_primero))

    st.markdown(
        f"**Segunda opción:** ODS {segundo}, {NOMBRES_ODS[segundo]} "
        f"({porcentaje(p_segundo)})"
    )

    if p_primero - p_segundo < MARGEN_ESTRECHO:
        st.info(
            "Los dos primeros objetivos están casi empatados, así que conviene leer el texto "
            "como **transversal** a ambos. En la evaluación, los textos con este margen fueron "
            "el 6% del total y concentraron el 31% de los errores."
        )

    with st.expander("Ver los cinco objetivos más probables"):
        for posicion in orden[:5]:
            ods = clases[posicion]
            st.progress(
                float(probabilidades[posicion]),
                text=f"ODS {ods}, {NOMBRES_ODS[ods]} ({porcentaje(probabilidades[posicion])})",
            )


def main() -> None:
    st.set_page_config(page_title="Clasificador de textos según los ODS", layout="centered")
    barra_lateral()

    st.title("Clasificador de textos según los ODS")
    st.write(
        "Escriba o pegue un párrafo en español y el sistema dice con cuál de los Objetivos de "
        "Desarrollo Sostenible de la Agenda 2030 se relaciona."
    )

    st.text_area(
        "Texto",
        key="texto",
        height=200,
        placeholder="Por ejemplo, un fragmento de un plan de desarrollo territorial, "
                    "un informe de política pública o una propuesta ciudadana.",
    )

    clasificar, ejemplo = st.columns([1, 1])
    pulsado = clasificar.button("Clasificar", type="primary", width="stretch")
    ejemplo.button("Cargar un ejemplo", on_click=poner_ejemplo, width="stretch")

    if not pulsado:
        return

    texto = st.session_state.texto.strip()
    if not texto:
        st.warning("Escriba un texto para clasificarlo.")
        return

    if len(texto.split()) < PALABRAS_MINIMAS:
        st.warning(
            f"El texto tiene menos de {PALABRAS_MINIMAS} palabras. El modelo aprendió de "
            "párrafos de unas 105 palabras, así que con textos muy cortos la predicción es "
            "poco confiable."
        )

    mostrar_resultado(texto, obtener_modelo())


main()
