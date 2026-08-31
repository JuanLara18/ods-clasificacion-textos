# -*- coding: utf-8 -*-
"""Prepara los dos archivos que se suben a Coursera.

    python scripts/exportar_entrega.py              # exporta el notebook tal como está
    python scripts/exportar_entrega.py --ejecutar   # lo corre de principio a fin primero

La entrega son dos archivos, el notebook con todas las celdas ejecutadas y su versión en
HTML. Quedan en `entrega/`, que no se versiona: es una copia derivada del notebook.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "Microproyecto_2.ipynb"
DIR_ENTREGA = RAIZ / "entrega"


def leer_argumentos():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nombre", default="Microproyecto2",
                        help="nombre base de los archivos de la entrega")
    parser.add_argument("--ejecutar", action="store_true",
                        help="corre el notebook completo antes de exportarlo")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="segundos máximos por celda al ejecutar")
    return parser.parse_args()


def correr(comando):
    print(" ".join(str(x) for x in comando))
    completado = subprocess.run(comando)
    if completado.returncode != 0:
        raise SystemExit(f"Falló: {' '.join(str(x) for x in comando)}")


def main():
    argumentos = leer_argumentos()
    if not NOTEBOOK.exists():
        raise SystemExit(f"No existe {NOTEBOOK}")
    DIR_ENTREGA.mkdir(exist_ok=True)

    if argumentos.ejecutar:
        correr([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", f"--ExecutePreprocessor.timeout={argumentos.timeout}", str(NOTEBOOK)])

    destino_ipynb = DIR_ENTREGA / f"{argumentos.nombre}.ipynb"
    shutil.copy2(NOTEBOOK, destino_ipynb)
    correr([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
            "--output", f"{argumentos.nombre}.html", "--output-dir", str(DIR_ENTREGA),
            str(NOTEBOOK)])

    print()
    print("Listo. Subir a Coursera estos dos archivos:")
    for archivo in sorted(DIR_ENTREGA.glob(f"{argumentos.nombre}.*")):
        print(f"  {archivo}  ({archivo.stat().st_size / 1024:.0f} KB)")
    print()
    print("Revisar antes de subir: que el HTML muestre la salida de todas las celdas y que "
          "las clasificaciones de los cuatro textos de prueba aparezcan al final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
