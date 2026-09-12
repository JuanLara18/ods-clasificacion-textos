# Micro proyecto 2: Alcanzando los objetivos de desarrollo sostenible

**Curso:** ML No Supervisado (2026-14)
**Fuente:** `Enunciado.pdf`
**Publicado:** semana 5
**Entrega: domingo 20 de septiembre de 2026, 11:59 PM (-05)**
**Peso:** 22,5% de la nota del curso
**Grupal** (*Graded Team Assignment* en Coursera)

El enunciado solo dice "al final de la semana 7". La fecha viene del gradebook de Coursera,
verificado el 17 de agosto de 2026 y registrado en
[`../Resumenes/00_Programa.md`](../Resumenes/00_Programa.md).

> **No correr `md2pdf.py` sobre este archivo sin `-o`.** El `Enunciado.pdf` de esta carpeta es
> el **original del curso**, no un PDF generado. Las notas en PDF van a `Enunciado_notas.pdf`.

## Contexto

La ONU adopta el 25 de septiembre de 2015 la Agenda 2030 para el desarrollo sostenible, con 17
**objetivos de desarrollo sostenible (ODS)** y 169 metas derivadas. Varias entidades hacen
seguimiento y evaluación de políticas públicas frente a esos objetivos. Es el caso del Fondo de
Población de las Naciones Unidas (UNFPA), que junto con instituciones públicas y herramientas
de participación ciudadana busca identificar problemas y evaluar soluciones, relacionando la
información con los ODS.

El cuello de botella es la **interpretación de información textual** proveniente de la
planeación participativa territorial: consume muchos recursos y exige expertos que relacionen
cada texto con los ODS. Automatizarlo permitiría orientar las políticas públicas con base en la
opinión de la población.

---

## A. Objetivo

Desarrollar una solución, basada en técnicas de procesamiento de lenguaje natural y machine
learning, que facilite la interpretación y el análisis de información textual para identificar
relaciones semánticas con los Objetivos de Desarrollo Sostenible.

---

## B. Conjunto de datos

Parte del **OSDG Community Dataset** (OSDG-CD), versión 2023: 40.067 textos en total, 3.000 de
ellos de fuentes relacionadas con Naciones Unidas, más documentos públicos, resúmenes de
artículos y reportes. El etiquetado lo hacen voluntarios de la plataforma, que validan la
relevancia de cada texto frente a un ODS según su conocimiento previo.

Los textos del proyecto fueron **traducidos al español** con herramientas tipo DeepL, y hubo
**aumentación de textos** con la API de ChatGPT.

Fuentes citadas en las notas al pie del PDF:

- OSDG Community Dataset: <https://osdg.ai/news/New-release-of-OSDG-Community-dataset>
- DeepL: <https://www.deepl.com/es/translator>
- API de ChatGPT: <https://chat.openai.com/g/g-I1XNbsyDK-api-docs>

El archivo se descarga desde Coursera y está en
[`data/Train_textosODS.xlsx`](data/Train_textosODS.xlsx). Lo que trae realmente, medido con
[`scripts/perfilar_corpus.py`](scripts/perfilar_corpus.py), está en las notas de lectura, más
abajo: **importa, porque no coincide del todo con lo que dice el enunciado**.

---

## C. Actividades para realizar

1. **Preparación de los textos** con bolsa de palabras (BOW) y pesado **TF-IDF**. Construir un
   **pipeline** que integre las transformaciones que se consideren adecuadas.
2. **Modelo de tópicos con LSA.** Sobre la matriz TF-IDF, aplicar **SVD truncado**
   (`TruncatedSVD` de scikit-learn). Explorar un número reducido de componentes (por ejemplo
   entre 10 y 20) y, para **al menos 5** de ellas, identificar y mostrar las palabras de mayor
   peso (*loadings*) a modo de tópicos. Interpretar cualitativamente si esos tópicos guardan
   relación con alguno de los 17 ODS.
3. **Modelo de clasificación** que relacione un texto con un ODS. Para manejar la
   dimensionalidad del espacio de entrada se puede reutilizar la misma descomposición SVD de la
   actividad anterior, o aplicar otra técnica de reducción que se considere pertinente.
4. **Evaluación** del modelo con textos que no hayan sido usados en el aprendizaje.

---

## D. Consideraciones

El algoritmo de clasificación y la técnica de reducción de la dimensionalidad **quedan a
elección del grupo, pero hay que justificar la elección**.

---

## E. Entregable

Notebook en `.ipynb` **y** `.html`, documentado con las justificaciones de cada paso y con
**las ejecuciones de cada celda visibles**. Debe mostrar las clasificaciones de **al menos
cuatro textos del conjunto de prueba**. Se adjuntan los dos archivos en el espacio de la
semana 7.

---

## F. Criterios de evaluación

| Actividad | Peso |
|---|---|
| Preparación de los datos, incluida la reducción de la dimensionalidad, justificando las decisiones | **30%** |
| Construcción del pipeline de preparación de datos | **15%** |
| Modelo de clasificación con búsqueda de hiperparámetros, validado con medidas de evaluación adecuadas. Se justifica la selección del algoritmo y de las métricas. Se aplica reducción de la dimensionalidad | **30%** |
| Evidencia del desempeño sobre textos no usados en el aprendizaje | **10%** |
| Modelo LSA sobre la matriz TF-IDF e interpretación cualitativa de al menos 5 tópicos frente a los ODS | **15%** |

---

## G. Bonificación adicional (opcional)

**15 puntos** por implementar el modelo en una aplicación interactiva con **Streamlit** que
permita ingresar texto libre, lo procese con el mismo pipeline del proyecto, devuelva el ODS
predicho y funcione correctamente. La asignación depende del funcionamiento y de la claridad de
la interacción. No reemplaza ningún criterio de la rúbrica.

---

# Notas de lectura

Lo que el enunciado no dice y hay que tener presente antes de escribir código.

## El corpus entregado no es el que describe el enunciado

`data/Train_textosODS.xlsx` trae **9.656 textos, no 40.067**: es el subconjunto de
entrenamiento traducido y aumentado, no el OSDG-CD completo. Dos columnas, `textos` y `ODS`,
sin nulos y sin textos duplicados.

**Las clases van del ODS 1 al 16.** El ODS 17 (*alianzas para lograr los objetivos*) **no
aparece**, así que el problema real es de **16 clases**, aunque el enunciado hable de 17. Hay
que decirlo explícitamente en el notebook: es exactamente el tipo de detalle que la rúbrica
premia en "justificando las decisiones tomadas".

Distribución de las etiquetas:

| ODS | textos | ODS | textos |
|---|---|---|---|
| 16 | 1.080 | 13 | 464 |
| 5 | 1.070 | 14 | 377 |
| 4 | 1.025 | 2 | 369 |
| 3 | 894 | 10 | 352 |
| 7 | 787 | 9 | 343 |
| 6 | 695 | 15 | 330 |
| 11 | 607 | 12 | 312 |
| 1 | 505 | 8 | 446 |

El desbalance es de **3,5 a 1** entre la clase mayor y la menor. No es extremo, pero basta para
que la **exactitud sea una métrica engañosa**: hay que reportar F1 macro, estratificar la
partición y la validación cruzada, y considerar `class_weight="balanced"`.

Los textos son párrafos: mediana de **105 palabras**, cuartiles en 82 y 135, rango de 24 a 268.
Es un tamaño cómodo para TF-IDF y suficiente para que el vocabulario de cada documento sea
informativo.

## Lo que exige la rúbrica y es fácil pasar por alto

- **El pipeline vale 15% aparte.** No basta con vectorizar: tiene que ser un `Pipeline` de
  scikit-learn que integre las transformaciones, para que el mismo objeto procese los textos
  nuevos. La bonificación de Streamlit depende de eso mismo.
- **La reducción de dimensionalidad se paga dos veces**, dentro del 30% de preparación y otra
  vez dentro del 30% del modelo. No es opcional, aunque un clasificador lineal funcione bien
  sobre TF-IDF disperso.
- **Búsqueda de hiperparámetros explícita.** El criterio la nombra: sin `GridSearchCV` o
  equivalente, con su justificación, se pierde parte del 30%.
- **Al menos cuatro textos de prueba clasificados** al final, mostrados uno por uno. Es un 10%
  que se pierde entero por olvido.
- **Al menos 5 componentes interpretadas** frente a los ODS, con sus palabras de mayor peso. La
  interpretación cualitativa es la parte que se califica, no la tabla de *loadings*.

## Sobre el español

`TfidfVectorizer` no trae lista de palabras vacías en español: `stop_words="english"` es la
única incluida. Hay que aportarla, y decidir explícitamente qué se hace con tildes, mayúsculas,
números y signos.

La aumentación con ChatGPT hacía temer variantes casi idénticas de un mismo texto repartidas a
lado y lado de la partición, que inflarían el desempeño. Medido: sobre una muestra de 2.000
documentos no hay **ningún** par con similitud coseno TF-IDF mayor o igual a 0,9 dentro de una
misma clase. La aumentación reescribió de verdad, no copió. Aun así conviene rehacer la
medición sobre el corpus completo antes de la entrega, y en todo caso la partición va
estratificada por clase.

## Relación con el material del curso

La semana 5 cubre exactamente esto: BOW y TF-IDF, embeddings, SVD y modelado de tópicos con sus
métricas, y está resumida en
[`../Resumenes/S05_Analisis_de_textos.md`](../Resumenes/S05_Analisis_de_textos.md), que es la
teoría de este proyecto. Los documentos originales están en
[`../Documentos/`](../Documentos/), prefijo `S05_`. La lectura
del capítulo 8 de Raschka (*Applying Machine Learning to Sentiment Analysis*) hace este mismo
recorrido sobre reseñas de IMDb, incluida la comparación entre LSA por SVD y LDA probabilístico,
y el libro está en [`../Libros/`](../Libros/README.md).
