# Cómo atacar el problema

Lo que encontramos midiendo, sobre los 9.656 textos del corpus, antes de escribir una línea del
método. La conclusión corta:

> **El enunciado plantea un problema de 16 clases excluyentes, y los datos no lo son.** La
> etiqueta única no es una propiedad de los textos, es una restricción del procedimiento con que
> los anotaron. Y el modelo lo nota: sus errores no son ruido, reproducen la estructura temática
> de la Agenda 2030.

Eso no cambia lo que hay que entregar, cambia **qué se mide y cómo se reporta**. Es el mismo giro
del micro proyecto 1, donde las medidas internas de validación preguntaban si había grupos
separados y una nube de píxeles resultó ser un continuo.

---

## 1. La línea base, y por qué es más alta de lo que parece

TF-IDF sin ajustar y una regresión logística, con validación cruzada estratificada de cinco
particiones sobre el corpus completo:

| | |
|---|---|
| Exactitud | **0,8886** |
| F1 macro | **0,8659** |

Para un problema de 16 clases con un desbalance de 3,5 a 1, ese es un punto de partida alto: la
clase mayoritaria sola daría 0,112. Conviene saberlo desde ya, porque fija la vara contra la cual
se van a comparar la SVD y el clasificador final. Si tras proyectar a 20 componentes el desempeño
cae mucho, hay que decirlo y explicar por qué se aplica igual, que es lo que pide la rúbrica.

## 2. Los errores no están repartidos al azar

Los quince pares que más se confunden, medidos como porcentaje de la clase real:

| Confusión | % | | Confusión | % |
|---|---|---|---|---|
| ODS 10 desigualdad → ODS 8 trabajo | **11,4** | | ODS 1 pobreza → ODS 10 desigualdad | 4,2 |
| ODS 10 desigualdad → ODS 1 pobreza | **8,2** | | ODS 8 → ODS 10 | 4,0 |
| ODS 9 industria → ODS 11 ciudades | 6,7 | | ODS 9 → ODS 8 | 3,8 |
| ODS 8 trabajo → ODS 5 género | 5,8 | | ODS 1 → ODS 8 | 3,8 |
| ODS 13 clima → ODS 7 energía | 5,6 | | ODS 8 → ODS 9 | 3,4 |
| ODS 9 industria → ODS 7 energía | 5,2 | | ODS 8 → ODS 1 | 3,4 |
| ODS 8 trabajo → ODS 4 educación | 5,2 | | ODS 2 hambre → ODS 3 salud | 3,3 |

Léase con atención lo que hay en esa lista. Desigualdad con trabajo y con pobreza. Industria con
ciudades y con energía. Clima con energía. Trabajo con género, que es la brecha salarial. Hambre
con salud, que es la nutrición. **No hay un solo par absurdo**: no aparece pobreza con vida
marina, ni educación con clima. El clasificador se equivoca exactamente donde un experto humano
dudaría.

Poniendo número a esa impresión, con las dos taxonomías estándar de la Agenda 2030:

| Agrupación | Errores por par dentro del mismo grupo | Entre grupos distintos | Razón |
|---|---|---|---|
| Las cinco P (People, Planet, Prosperity, Peace, Partnership) | 7,0 | 3,6 | **1,93** |
| Pastel de bodas (biosfera, sociedad, economía) | 5,5 | 4,0 | 1,36 |

Un par de ODS de la misma P recibe casi el doble de errores que un par de P distintas.

## 3. La estructura de la Agenda 2030 emerge de los errores

Esto es lo que da vuelta al problema. Tomando la matriz de confusión, simetrizándola y tratando
"cuánto se confunden dos ODS" como una medida de similitud, se puede agrupar los dieciséis
objetivos **sin usar sus etiquetas ni sus definiciones**, solo con los errores del modelo. Con
tres grupos sale esto:

| Grupo | ODS |
|---|---|
| **Asuntos humanos e instituciones** | 1 pobreza, 2 hambre, 3 salud, 4 educación, 5 género, 8 trabajo, 10 desigualdad, 16 paz |
| **Sistemas técnicos y ambientales** | 7 energía, 9 industria, 11 ciudades, 12 consumo, 13 clima |
| **Ecosistemas naturales** | 6 agua, 14 vida marina, 15 vida terrestre |

Con cuatro grupos, el que se desprende solo es **ODS 16, paz y justicia**, que es exactamente su
propia P en la taxonomía oficial de Naciones Unidas.

Nadie le dijo al modelo que existen las cinco P ni el pastel de bodas del Stockholm Resilience
Centre. La estructura está en el lenguaje de los textos, y el modelo la recupera por la puerta de
atrás, equivocándose. La figura está en
[`../results/estructura_de_los_errores.png`](../results/estructura_de_los_errores.png).

## 4. Lo que encontramos fuera del machine learning

### La etiqueta única es un artefacto de la anotación

El OSDG Community Dataset, del que sale nuestro corpus, documenta su procedimiento sin ambages:

> *"All texts are validated against only one associated SDG label."* Y sobre pedir varias
> etiquetas a la vez: *"volunteers are never asked to assign multiple SDGs to a single text, as
> this approach is considered extremely inefficient"*.

Es decir: **a los anotadores nunca se les dio la opción de decir que un texto toca dos objetivos**.
La monoetiqueta no describe los textos, describe la interfaz con que se recogieron. Un texto sobre
empleo juvenil femenino en zonas rurales es ODS 5, ODS 8 y ODS 10 a la vez, y el dataset se ve
obligado a elegir uno.

El dataset original trae además tres columnas que **nuestro subconjunto perdió**:
`labels_positive`, `labels_negative` y `agreement`, esta última definida como

$$\text{agreement}=\frac{\lvert \text{positivos}-\text{negativos}\rvert}{\text{positivos}+\text{negativos}}$$

sobre un mínimo de tres voluntarios por texto. Es decir, en la fuente **cada etiqueta viene con una
medida de cuánto dudaron los humanos**, y nosotros recibimos solo la etiqueta ganadora. Los 32.431
excerpts del original acumulan 217.147 validaciones: casi siete votos por texto.

### Los ODS están diseñados para solaparse

La literatura de política pública lleva una década midiéndolo. El marco de referencia es
**Nilsson, Griggs y Visbeck (2016)**, que puntúa las interacciones entre objetivos en una escala
de siete puntos, de $+3$ *indivisible* a $-3$ *cancelling*, pasando por *reinforcing*,
*enabling*, *consistent*, *constraining* y *counteracting*. Es el instrumento estándar del campo,
usado en 27 de los estudios de una revisión reciente.

La consecuencia para nosotros es directa: **si los objetivos son indivisibles por diseño, pedirle
a un clasificador que los separe con una frontera dura es pedirle algo que el dominio no
sostiene**. Y la literatura de clasificación automática de ODS lo reconoce: los sistemas
existentes difieren mucho entre sí, tienen sesgos sistemáticos por objetivo, y el problema se
describe explícitamente como multietiqueta, donde *"accuracy cannot be relied upon to determine
model performance"*.

### La ausencia del ODS 17 deja de ser rara

En el pastel de bodas y en las cinco P, el ODS 17 (alianzas) **no es un tema, es el medio de
implementación de todos los demás**: atraviesa la estructura en vez de ocupar un lugar en ella.
Que no haya textos suyos en el corpus no es un descuido, es coherente con que no sea una categoría
temática comparable a las otras dieciséis.

## 5. La evidencia decisiva: el modelo sabe cuándo duda

Si la tesis es cierta, los errores deberían ser dudas y no disparates. Se mide mirando en qué
posición queda el ODS correcto cuando el modelo no lo pone primero:

| | Exactitud acumulada |
|---|---|
| top-1 | 0,8886 |
| **top-2** | **0,9572** |
| top-3 | 0,9777 |
| top-5 | 0,9899 |

**De los 1.076 errores, en 663 el ODS correcto quedó en segunda posición.** Es el 62% de los
fallos. Y la confianza acompaña:

| | Confianza media de la clase elegida |
|---|---|
| Cuando acierta | 0,774 |
| Cuando falla | 0,433 |

En los fallos, la diferencia media entre la probabilidad de la clase elegida y la de la correcta
es de 0,29, y en **331 de los 1.076 fallos esa diferencia es menor que 0,10**: un empate técnico.
El modelo no se equivoca con seguridad, se equivoca dudando. Un tercio de sus errores son casos en
los que dos objetivos le parecían casi igual de plausibles, que es precisamente lo que pasa cuando
el texto habla de los dos.

---

## 6. Lo que proponemos hacer con esto

Cinco ideas, ordenadas por lo que cuestan frente a lo que aportan. Ninguna reemplaza lo que exige
la rúbrica; todas se montan encima.

### Idea 1. Una medida de error que sepa de qué habla

Es el equivalente del $\Delta E_{00}$ del micro proyecto 1. La exactitud castiga igual confundir
desigualdad con trabajo que confundir desigualdad con vida marina, y el dominio dice que no son
comparables. Se define una matriz de costo $c(a,b)$ entre objetivos, derivada de la estructura de
la Agenda 2030, y se reporta el **costo medio del error** junto al F1 macro.

La versión barata usa las cinco P: costo 0 si acierta, 1 si falla dentro de la misma P, 2 si falla
entre P distintas. La versión cara usa la escala de Nilsson. Con la barata ya se puede decir algo
que ninguna métrica estándar dice: qué fracción del error del modelo es **error temático real** y
qué fracción es **ambigüedad legítima del dominio**.

**Cuesta:** unas veinte líneas. **Aporta:** el argumento entero de la sección de justificación, que
vale 30%.

### Idea 2. Entregar dos objetivos, no uno

El destinatario del enunciado es el UNFPA, que quiere aliviar el trabajo de expertos que leen
textos y los relacionan con los ODS. Para ese usuario, un sistema que propone **los dos ODS más
probables con su confianza** y acierta el 95,7% de las veces es más útil que uno que propone uno
solo y acierta el 88,9%. Se reporta el top-2 al lado del top-1, y se explica por qué.

**Cuesta:** dos líneas y un párrafo. **Aporta:** convierte un número peor en un producto mejor, y
alimenta la app de Streamlit de la bonificación.

### Idea 3. Abstención cuando el empate es real

Cuando el margen entre el primer y el segundo objetivo es menor que un umbral, el sistema no
elige: devuelve los dos y marca el texto como **transversal**. Con umbral 0,10 eso ocurriría en
alrededor del 3,4% de los textos, y ahí se concentra un tercio de los errores actuales. Es una
regla de decisión, no un modelo nuevo.

**Cuesta:** poco. **Aporta:** es la forma honesta de tratar un problema multietiqueta con datos
monoetiqueta, y da material de análisis de sobra.

### Idea 4. Leer los tópicos de LSA contra la estructura, no contra los ODS

El criterio del 15% pide interpretar al menos cinco componentes frente a los 17 objetivos. La
expectativa ingenua es que cada componente corresponda a un ODS. **Probablemente no va a pasar**,
y ahora sabemos por qué: la estructura que hay en el lenguaje tiene tres o cinco bloques, no
dieciséis. Si las primeras componentes se alinean con los bloques que salieron de la sección 3,
eso no es un fracaso de la interpretación, es **la confirmación por una segunda vía** de que la
estructura del corpus es de bloques temáticos.

Se puede medir en vez de afirmarlo: proyectar los centroides de cada ODS sobre las componentes y
mirar qué agrupa cada una.

**Cuesta:** nada extra, es la actividad 2 del enunciado bien hecha. **Aporta:** el 15% con un
argumento propio en vez de una lista de palabras comentada.

### Idea 5. Recuperar el `agreement` perdido

La más ambiciosa y la única que puede no salir. Los textos originales del OSDG-CD están en inglés
y los nuestros traducidos, así que no se pueden cruzar por texto exacto. Pero sí se podría estimar,
sobre una muestra, qué fracción de nuestros textos corresponde a etiquetas de bajo acuerdo entre
voluntarios. Si los textos donde el modelo falla resultan ser los mismos donde los humanos no se
pusieron de acuerdo, **el techo de desempeño alcanzable queda medido**, y se acaba la discusión
sobre si el modelo se puede mejorar más.

**Cuesta:** una tarde, con riesgo de no llegar a nada. **Aporta:** si sale, es el resultado más
fuerte del proyecto. Va al final, si sobra tiempo, y no antes de tener la rúbrica cubierta.

---

## 7. Lo que esto no cambia

Sigue habiendo que entregar un pipeline de scikit-learn, una SVD truncada con sus tópicos
interpretados, un clasificador con búsqueda de hiperparámetros y cuatro textos de prueba
clasificados. Nada de lo anterior sustituye un solo criterio de la rúbrica: se reporta **además**
de F1 macro y matriz de confusión, no en lugar de.

Y hay un riesgo que vigilar: el corpus está **traducido automáticamente y aumentado con un modelo
de lenguaje**. Parte de la estructura que observamos podría venir del traductor y no de los textos
originales. No lo podemos descartar con los datos que tenemos, y conviene decirlo en el notebook
antes de que lo diga el calificador.

---

## Referencias

- **OSDG Community Dataset (OSDG-CD)**, Zenodo y
  [github.com/osdg-ai/osdg-data](https://github.com/osdg-ai/osdg-data). Procedimiento de
  anotación, columnas `labels_positive`, `labels_negative` y `agreement`, y la decisión explícita
  de validar cada texto contra un solo ODS.
- **Nilsson, M., Griggs, D. y Visbeck, M. (2016).** *Map the interactions between Sustainable
  Development Goals*. La escala de siete puntos, de indivisible a cancelling. Marco de referencia
  del [International Science Council](https://council.science/publications/working-paper-a-draft-framework-for-understanding-sdg-interactions-2016/).
- **Stockholm Resilience Centre (2016).** [*The SDGs wedding cake*](https://www.stockholmresilience.org/research/research-news/2016-06-14-the-sdgs-wedding-cake.html).
  Los ODS como tres capas anidadas, con la biosfera como base.
- **Naciones Unidas.** Las cinco P de la Agenda 2030: People, Planet, Prosperity, Peace,
  Partnership.
- **Revisión de sistemas automáticos de etiquetado de ODS**, sobre los sesgos por objetivo y el
  carácter multietiqueta del problema:
  [PMC11366727](https://pmc.ncbi.nlm.nih.gov/articles/PMC11366727/).
