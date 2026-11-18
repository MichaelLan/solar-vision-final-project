# Entrenamiento y evaluación

El dataset se obtiene del repositorio original de [Raptor Maps](https://github.com/RaptorMaps/InfraredSolarModules). El ZIP no se versiona. Se usan cuatro clases: No-Anomaly, Cell, Cell-Multi y Diode.

## Partición

Con semilla 42 se seleccionan, por clase, 1000 imágenes de entrenamiento, 100 de validación y 150 de prueba. Antes del muestreo se eliminan duplicados exactos de píxeles. Después se aplica una deduplicación global: una imagen se descarta si su distancia RMS de píxeles (escala de grises BT.601 en [0, 1]) a cualquier imagen ya seleccionada es ≤ 0,02, sin importar la clase ni la partición. La prueba se selecciona primero y **solo** con imágenes que nunca figuraron en el manifiesto histórico (`../artifacts/historical/manifest.json`). Las imágenes históricas pueden formar parte del entrenamiento o la validación. El criterio anterior, dHash ≤ 2, se reemplazó porque los pares con dHash ≤ 2 tenían una distancia de píxeles similar a la de pares aleatorios, así que no detectaba duplicados reales. `manifest.json` registra el método, el umbral, los conteos y los rechazos.

## Protocolo

El protocolo queda en `metrics.json` (`protocol`) antes de entrenar. Los candidatos son:

- regresión logística (C=1, sobre píxeles crudos; línea base);
- CNN w8 y CNN w16: tres convoluciones 3×3, ReLU, max-pooling y pooling promedio global, más una capa densa. La entrada se normaliza por imagen, el entrenamiento usa volteos aleatorios, AdamW y OneCycle durante 40 épocas, sin early stopping. Cada CNN se entrena con las semillas 42, 43 y 44.

Se selecciona la arquitectura con mayor F1 macro medio en validación; si w8 queda a 0,01 o menos de w16, se elige w8. Se despliega la ejecución con semilla 42. La prueba nueva se evalúa una sola vez, solo con el modelo seleccionado y la línea base.

Las métricas de prueba incluyen intervalos bootstrap del 95 % (1000 remuestreos), sensibilidad y especificidad de anomalía frente a normal, y una lista de errores.

## Ejecución

Desde `ml/`:

```bash
uv sync --locked          # incluye torch CPU (grupo train)
PYTHONPATH=src uv run --no-sync python -m solar_ml.cli --archive ../dataset/2020-02-14_InfraredSolarModules.zip
uv run --no-sync pytest -q
PYTHONPATH=src uv run --no-sync python -m solar_ml.verify_backend --archive ../dataset/2020-02-14_InfraredSolarModules.zip
```

`solar_ml.verify_backend` vuelve a calcular la paridad entre `backend/src/infra/cnn.py` (`ConvClassifier`) y `solar_ml.export.forward` sobre todo el conjunto de prueba. Solo reescribe el bloque `selected_model.backend_convclassifier_parity` de `metrics.json`; no reentrena ni modifica otras métricas. Debe ejecutarse después de entrenar.

## Salidas

La ejecución escribe en `../artifacts/`:

- `manifest.json`, `dataset_info.json` y `metrics.json`;
- `model.json`: contrato 2.0.0, `architecture: "cnn"` y `sha256` de las capas y la entrada;
- `baseline_model.json`: regresión logística, `architecture: "dense"`;
- `reference_prediction.json`;
- figuras en `../artifacts/figures/`.

`solar_ml/export.py` contiene el pase hacia adelante de referencia en numpy. El navegador entrega 960 valores en [0, 1] por filas (`floor(0.299 R + 0.587 G + 0.114 B + 0.5) / 255`, 24 × 40). La normalización por imagen, `(x − media) / (std poblacional + 1e-6)`, la aplica el servidor según `input.normalization`.

Los resultados anteriores (`../artifacts/historical/`) usaban otra partición y un conjunto de prueba ya observado, por lo que no son comparables con los actuales.
