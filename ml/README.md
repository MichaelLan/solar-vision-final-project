# Entrenamiento y evaluación

El dataset se obtiene del repositorio original de [Raptor Maps](https://github.com/RaptorMaps/InfraredSolarModules). El ZIP original no se versiona. El pipeline valida el archivo ZIP y `module_metadata.json`, comprueba imágenes y dimensiones, elimina duplicados exactos, selecciona 400 imágenes por clase sin parejas dHash de distancia Hamming ≤ 2 entre las cuatro clases y genera un manifiesto reproducible. El orden de selección prioriza Diode, Cell-Multi, Cell y No-Anomaly porque las anomalías minoritarias tienen menos patrones dHash distintos. Si la fuente cambia o la selección no alcanza 400 por clase, el proceso falla explícitamente.

Desde `ml/`:

```bash
uv sync --dev
PYTHONPATH=src uv run --no-sync python -m solar_ml.cli --download
```

Si la descarga automática no está disponible, descargue `2020-02-14_InfraredSolarModules.zip` manualmente del repositorio original y ejecute:

```bash
PYTHONPATH=src uv run --no-sync python -m solar_ml.cli --archive /ruta/al/2020-02-14_InfraredSolarModules.zip
```

El entrenamiento crea `../artifacts/model.json`, `metrics.json`, `manifest.json` y `dataset_info.json`, además de las figuras en `../report/figures/`. También exporta dos imágenes por clase del conjunto de prueba a `demo/`, con `provenance.json` para facilitar la grabación. La selección se basa exclusivamente en F1 macro del conjunto de validación. El conjunto de prueba se consulta una vez para el modelo seleccionado. El modelo exportado usa los mismos pesos que el evaluado. El navegador debe decodificar la imagen de 24 × 40 píxeles (ancho × alto verificados en el ZIP original) y entregar 960 valores por filas usando `floor(0.299 R + 0.587 G + 0.114 B + 0.5) / 255`; no hay redimensionamiento.

La selección greedy dHash se realiza antes de la partición, de modo que ninguna pareja seleccionada tenga distancia Hamming ≤ 2, incluso entre clases diferentes. Reducimos la muestra inicial de 1.000 a 400 por clase porque una selección global de 1.000 sin semejanzas dHash no era viable con las clases minoritarias. El manifiesto comprueba pares cercanos entre particiones y conserva el hash del tensor gris y del tensor cuantizado a 16 niveles. No se garantiza independencia por sitio o instalación: el dataset no provee ese identificador. Diferencias de decodificación JPEG entre Pillow y el navegador pueden ocasionar cambios mínimos en tensores; se deben comprobar con imágenes reales al ejecutar el experimento.
