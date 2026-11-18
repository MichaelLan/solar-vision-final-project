# Solar Vision

Clasificación académica de cuatro patrones en imágenes térmicas de módulos fotovoltaicos (No-Anomaly, Cell, Cell-Multi y Diode) con el conjunto InfraredSolarModules de Raptor Maps. El repositorio contiene la preparación de datos y el entrenamiento (`ml/`), una API FastAPI para Cloudflare Workers (`backend/`), una interfaz Vue 3 (`frontend/`) y un informe IEEE en Typst (`report/`). La herramienta apoya una primera revisión visual; no emite diagnósticos eléctricos.

## Estado

- **Modelo actual (2.0.0):** CNN compacta (`cnn_w16`, 23 556 parámetros), seleccionada por F1 macro medio de validación sobre tres semillas, según un protocolo registrado antes de entrenar (`artifacts/metrics.json`, clave `protocol`).
- **Evaluación:** conjunto de prueba nuevo de 600 imágenes (150 por clase), sin imágenes de la versión anterior, evaluado una sola vez.
- **Despliegue:** la URL pública [solar-vision.michaellan-eng.workers.dev](https://solar-vision.michaellan-eng.workers.dev/) sirve el modelo 2.0.0 desde el 18 de noviembre de 2026. Se verificó que `/api/model/info` devuelve el mismo SHA-256 que `artifacts/model.json` y que `POST /api/predict` con el tensor de referencia (imagen 3578) reproduce `artifacts/reference_prediction.json` (Cell-Multi 0,678). La versión anterior (1.0.0, regresión logística) ya no está publicada.

### Resultados en la prueba nueva

| Modelo | Exactitud | F1 macro | AUC macro (OvR) | Sensibilidad anomalía | Especificidad |
|---|---|---|---|---|---|
| CNN w16 (desplegada) | 0,757 [0,720; 0,790] | 0,748 [0,713; 0,780] | 0,929 [0,914; 0,941] | 0,896 | 0,913 |
| Regresión logística (línea base) | 0,582 | 0,575 | 0,818 | 0,849 | 0,587 |

Los intervalos son bootstrap percentiles del 95 % (1000 remuestreos). F1 por clase de la CNN: Diode 0,945, No-Anomaly 0,820, Cell 0,627 y Cell-Multi 0,601. La confusión principal es Cell ↔ Cell-Multi. El experimento anterior (`artifacts/historical/`, exactitud 0,5625) usaba otra partición y un conjunto de prueba ya consultado, por lo que no es comparable. Detalles en el [informe PDF](report/informe.pdf) y en [ml/README.md](ml/README.md).

## Arquitectura

```text
dataset/2020-02-14_InfraredSolarModules.zip → ml/ (partición, entrenamiento, evaluación)
                         ↓ artifacts/model.json, metrics.json, manifest.json, figures/
imagen → navegador Vue (valida, luminancia 24×40) → POST /api/predict (960 valores en [0,1])
                                                       ↓
     Python Worker / FastAPI → estandarización por imagen → CNN en numpy → clase y probabilidades
```

El navegador acepta PNG, JPEG o WebP de **24 × 40 píxeles** (sin redimensionar), hasta 5 MB por archivo y hasta 20 por lote. Calcula 960 valores por filas: `floor(0.299 R + 0.587 G + 0.114 B + 0.5) / 255`. El servidor valida el vector, aplica `(x − media) / (std + 1e-6)` por imagen y ejecuta el paso hacia adelante de la CNN implementado en numpy a partir de `artifacts/model.json` (PyTorch no está disponible en Workers). En las 600 imágenes de prueba, la diferencia máxima de probabilidad frente a PyTorch es 3,2 × 10⁻⁷ y las predicciones coinciden al 100 %. No se almacenan imágenes.

Rutas: `GET /api/health`, `GET /api/model/info` (versión, arquitectura, SHA-256 y contrato de entrada) y `POST /api/predict` con `{"pixels": [960 valores], "width": 24, "height": 40}`.

## Requisitos

- `uv`; Python 3.12 para `ml/` (incluye PyTorch CPU) y Python 3.14 para `backend/`.
- Node.js 24 y pnpm 12 para `frontend/`.
- Typst reciente con `@preview/charged-ieee:0.1.4` para el informe.
- ZIP original `2020-02-14_InfraredSolarModules.zip` de [Raptor Maps](https://github.com/RaptorMaps/InfraredSolarModules) (licencia MIT) en `dataset/`; no se versiona.

Los lockfiles de `backend/`, `ml/` y `frontend/` están versionados. El lockfile de pnpm 12 contiene dos documentos YAML por diseño; no deben separarse.

## Preparar, entrenar y evaluar

```bash
cd ml
uv sync --python 3.12 --locked
PYTHONPATH=src uv run --no-sync python -m solar_ml.cli --archive ../dataset/2020-02-14_InfraredSolarModules.zip
```

El pipeline usa semilla 42: elimina duplicados exactos, aplica una deduplicación global por distancia RMS de píxeles (≤ 0,02) entre clases y particiones, selecciona 1000/100/150 imágenes por clase (entrenamiento/validación/prueba), entrena la regresión logística y las CNN w8 y w16 con las semillas 42, 43 y 44, selecciona según el protocolo y evalúa la prueba una sola vez. Escribe los artefactos en `artifacts/` y las figuras en `artifacts/figures/`. Ver [ml/README.md](ml/README.md).

## Ejecutar API y frontend localmente

```bash
python3 backend/scripts/package_model.py   # incrusta artifacts/model.json en el Worker
cd backend
uv sync --python 3.14 --locked --dev
uv run --no-sync uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

En otra terminal:

```bash
cd frontend
pnpm install --frozen-lockfile --ignore-scripts
pnpm dev
```

Vite redirige `/api` a `localhost:8000`. Compruebe `http://127.0.0.1:8000/api/health` y `/api/model/info` (debe indicar la versión 2.0.0) antes de enviar imágenes. Para probar, use las imágenes de `ml/demo/`, tomadas del conjunto de prueba (procedencia en `ml/demo/provenance.json`).

`--ignore-scripts` evita los scripts de instalación de las dependencias. `pnpm dev`, `pnpm test` y `pnpm build` sí ejecutan código JavaScript; ejecútelos solo en un entorno autorizado, como el CI aislado.

## Pruebas e informe

```bash
cd ml && PYTHONPATH=src uv run --no-sync pytest -q && uv run --no-sync ruff check src tests
cd ../backend && uv run --no-sync pytest -q tests && uv run --no-sync ruff check src tests
cd ../frontend && pnpm test && pnpm build
cd ..
python3 report/build_report.py
typst compile --root . report/main.typ report/informe.pdf
```

`report/build_report.py` usa solo la biblioteca estándar: genera `report/results.typ` con todas las cifras del informe a partir de `artifacts/*.json` y falla si alguna afirmación cualitativa del texto deja de cumplirse. Con `--figures` (requiere matplotlib y Pillow, por ejemplo `ml/.venv/bin/python report/build_report.py --figures`) también regenera las figuras de `report/figures/` desde las métricas y el ZIP. La captura de la interfaz (`report/figures/app_result.png`) procede de la aplicación pública con el modelo 2.0.0, recortada para que sea legible a una columna. El guion del [video demostrativo](docs/video.md) dura menos de diez minutos.

## Cloudflare Workers

La configuración [Wrangler](backend/wrangler.jsonc) empaqueta la API Python (FastAPI y numpy) y sirve `frontend/dist` mediante Workers Static Assets.

```bash
cd backend
uv run --no-sync pywrangler deploy --dry-run
uv run --no-sync pywrangler dev
```

La publicación se hace con [GitHub Actions](docs/ci-cloudflare.md): el workflow verifica ML, backend, frontend e informe y, en `main`, despliega y comprueba la URL pública. Tras cada despliegue, verifique que `/api/model/info` devuelva la versión y el SHA-256 de `artifacts/model.json`. No guarde tokens ni credenciales en Git.

## Límites

La partición equilibrada no representa la frecuencia real de las clases (No-Anomaly es el 50 % del conjunto completo y el 68 % de las cuatro clases elegidas). El conjunto no tiene identificadores de planta, por lo que la evaluación no mide la generalización a otras instalaciones. A 24 × 40 píxeles es difícil distinguir una celda caliente de varias. Solo se modelan cuatro de las doce clases, y las imágenes de otras categorías se asignarán a una de ellas. Las probabilidades no están calibradas. El clasificador no localiza anomalías ni estima temperatura.
