# AGENTS.md

Guía para trabajar en este repositorio, humano o agente. El contexto del proyecto está en
[`README.md`](README.md), la rúbrica y las notas de lectura del corpus en
[`Enunciado.md`](Enunciado.md), y las decisiones cerradas en
[`docs/decisiones.md`](docs/decisiones.md).

Es el repositorio del **micro proyecto 2**. El micro proyecto 1, sobre paletas de color, vive
aparte en `../Proyecto/` y no comparte nada con este: cada uno tiene su remoto y su entrega.

## Lo primero

El entregable es `notebooks/Microproyecto_2.ipynb` y se califica con la rúbrica del enunciado.
Leerla antes de escribir algo: cada sección del notebook corresponde a un criterio con su
porcentaje, y el trabajo se mide contra eso y no contra lo que sería técnicamente más elegante.

Tres exigencias literales de la rúbrica que no se negocian:

1. **Un `Pipeline` de scikit-learn de verdad**, que integre las transformaciones. Vale 15% por
   sí solo, y es lo que permite que la app de Streamlit procese un texto nuevo igual que el
   entrenamiento. Vectorizar a mano en celdas sueltas pierde ese criterio.
2. **Reducción de dimensionalidad aplicada**, aunque un clasificador lineal sobre TF-IDF
   disperso funcione bien sin ella. Se paga dentro del 30% de preparación y otra vez dentro del
   30% del modelo.
3. **Búsqueda de hiperparámetros explícita**, con su justificación, y **al menos cuatro textos
   de prueba clasificados** al final. Ese último punto es un 10% que se pierde entero por
   olvido.

## Lo que el corpus obliga

Medido con `scripts/perfilar_corpus.py`, no supuesto:

- **16 clases, no 17.** El ODS 17 no está en el archivo. Decirlo en el notebook, no callarlo.
- **Desbalance de 3,5 a 1.** La métrica principal es **F1 macro**; la exactitud se reporta al
  lado pero no decide nada. Partición y validación cruzada **estratificadas**. Vale considerar
  `class_weight="balanced"` y justificar la decisión en cualquier sentido.
- **Español.** `TfidfVectorizer` no trae palabras vacías en español: hay que aportar la lista y
  decidir explícitamente qué se hace con tildes, mayúsculas, números y signos.

## Dónde va cada cosa

- **El método, entero, en el notebook.** `notebooks/Microproyecto_2.ipynb` es autocontenido y no
  importa nada nuestro. El calificador recibe dos archivos y no este repositorio, así que un
  notebook que importe de `src/` no corre para nadie más.
- **Lo que no es el método, en `scripts/`.** Perfilar el corpus y exportar la entrega. Los
  scripts no importan del notebook ni el notebook de ellos.
- **Decisiones: parámetros de los estimadores, no un diccionario de configuración.** Cada
  parámetro vive donde se usa, con su valor por defecto, y se argumenta en la sección donde
  aparece. Si aparece un número discutible, su argumento va a `docs/decisiones.md`.

## Reglas del notebook

1. Se edita de a una persona a la vez, porque es JSON y los conflictos son ilegibles.
2. Durante el desarrollo se guarda **sin salidas**:
   `jupyter nbconvert --clear-output --inplace notebooks/Microproyecto_2.ipynb`.
   Solo la corrida final va con salidas, porque el enunciado exige ejecuciones visibles.
3. No se agregan celdas de exploración al entregable. Lo exploratorio va a un script o a un
   notebook aparte que no se versiona.
4. Cada función se define en la sección que la explica. El orden de las celdas es el de los
   criterios de evaluación.
5. **El notebook es la única fuente.** Se edita celda por celda; no hay script que lo genere.
6. **Sin diagramas dentro del notebook.** Un bloque ```mermaid lo dibuja JupyterLab, pero el
   HTML exportado lo pide a un CDN y sin red desaparece. Los diagramas viven en los `.md`.
7. **Nada de referencias a lo que no viaja con la entrega.** El notebook no nombra los scripts,
   ni este repositorio, ni rutas suyas. El corpus se lee de una carpeta `data/` que el notebook
   busca a su lado o un nivel más arriba.
8. **Semilla fija en todo lo que la acepte**, partición incluida, y declarada. El notebook se
   corre completo antes de entregar y tiene que dar lo mismo.

## Estilo de la escritura

- **Sin guiones ni rayas como puntuación en prosa**, ni `-` suelto entre espacios, ni `--`, ni
  `—`, ni `–`. Se reescribe con coma, dos puntos o paréntesis. No aplica a la sintaxis
  obligatoria: separadores de tablas, banderas de línea de comandos, identificadores.
- **Sin emojis**, en ningún archivo.
- Prosa corrida, en primera persona del plural, explicando el mecanismo y no solo el resultado.
  Antes de un número va la frase que dice qué se está calculando, y después la que lo interpreta.
- Nombres de funciones, variables y módulos en español sin tildes. Docstrings en español.
- Commits de una línea, asunto convencional, sin cuerpo y sin trailer de coautoría.

## Nunca

- Crear un venv, `__pycache__` o archivos `.pyc` dentro del Drive. El entorno del bimestre
  vive fuera de él, en `%USERPROFILE%\Envs\miad-2026-14`.
- Versionar `entrega/`: es copia derivada del notebook y se regenera con
  `scripts/exportar_entrega.py`.
- Versionar el enunciado en PDF, los PDF de `docs/` ni `docs/plan.md`. El primero es material
  del curso, los segundos se regeneran con `md2pdf.py` y el tercero es logística nuestra: los
  tres viven en el Drive y el `.gitignore` los deja fuera.
- Correr `md2pdf.py` sobre `Enunciado.md` sin `-o`: el PDF hermano es el original del curso y se
  sobrescribiría. Las notas en PDF viven en `Enunciado_notas.pdf`.
- Tocar las etiquetas del corpus para "arreglar" un resultado. Si algo del conjunto estorba, se
  discute y se deja constancia en la bitácora de `docs/decisiones.md`.

## Antes de dar algo por terminado

```bash
python scripts/perfilar_corpus.py       # el corpus sigue siendo el que creemos
```

Y correr el notebook completo revisando tres cosas: que la matriz de confusión no tenga una
clase entera en cero, que el F1 macro no esté muy por debajo de la exactitud (señal de que las
clases pequeñas se están sacrificando) y que los cinco tópicos interpretados sigan diciendo algo
reconocible, porque las componentes cambian con cualquier ajuste del vectorizador.
