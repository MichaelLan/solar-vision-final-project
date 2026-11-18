# Solar Vision

Clasificación académica de cuatro patrones en imágenes térmicas de módulos fotovoltaicos. El repositorio contiene preparación y evaluación de datos, API FastAPI diseñada para Cloudflare Workers, interfaz Vue 3 y un informe IEEE en Typst. La herramienta apoya una primera revisión visual; no emite diagnósticos eléctricos.

## Estado verificado

Al 17 de noviembre de 2026, el ZIP original está disponible en `dataset/`, se verificaron sus 20.000 metadatos y se entrenó y evaluó el modelo final. El artefacto de pesos está en `artifacts/model.json` y se empaquetó para el backend. La API ASGI local respondió 200 a una imagen real del conjunto de prueba. Se instaló pywrangler 1.17.9 desde caché y `pywrangler deploy --dry-run` generó un bundle real de 9,666 MiB. **La entrega aún es parcial:** `pywrangler dev` no pudo arrancar por la restricción `uv_interface_addresses` del sandbox, y no hay despliegue público verificado. Existen `backend/uv.lock` y `backend/pylock.toml`; faltan `ml/uv.lock` y `frontend/pnpm-lock.yaml`, que GitHub Actions exige antes de desplegar. El PDF presenta métricas reales y declara los límites del despliegue. Véase [la prueba de viabilidad](backend/feasibility/README.md) y la [guía de CI y Cloudflare](docs/ci-cloudflare.md).

Las pruebas del backend (8/8) y del pipeline ML (6/6) pasaron en el entorno local; la compilación de Typst también pasó. Una prueba HTTP ASGI con un JPEG real devolvió 200 y probabilidades coincidentes con la referencia ML. Entre las imágenes demostrativas hay un error observable: Cell-Multi 3557 se clasificó como Cell. La latencia del prototipo corresponde solo a CPython con pesos representativos, no a Cloudflare. El build y las pruebas del frontend requieren verificación tras instalar dependencias con pnpm.

## Arquitectura

```text
dataset/2020-02-14_InfraredSolarModules.zip → ml/ (preparación, split, entrenamiento, evaluación)
                              ↓ artifacts/model.json, metrics.json y figuras
imagen → navegador Vue (decodificación, luminancia 24×40) → POST /api/predict
                                                       ↓
                         Python Worker / FastAPI → pesos empaquetados → clase y probabilidades
```

El navegador acepta PNG, JPEG o WebP de **24 píxeles de ancho por 40 de alto**, sin redimensionar. Calcula 960 valores en orden fila: `floor(0.299 R + 0.587 G + 0.114 B + 0.5) / 255`. La API valida dimensiones, longitud, rango finito y tamaño de solicitud. Las capas de aplicación dependen de un contrato de clasificación; la implementación de pesos vive en infraestructura. No se almacenan imágenes subidas. El preprocesamiento del navegador y Pillow deben compararse con JPEG reales antes del despliegue.

## Requisitos

- Python 3.12 para `ml/` y Python 3.14 para `backend/` (instalables mediante `uv`).
- `uv`, Node.js, pnpm y Typst reciente con `@preview/charged-ieee:0.1.4`.
- Conexión a los registros de paquetes y Cloudflare para instalar y desplegar.
- ZIP original `2020-02-14_InfraredSolarModules.zip` de [Raptor Maps](https://github.com/RaptorMaps/InfraredSolarModules). La copia local está en `dataset/` y no se versiona.

Los lockfiles de `backend/`, `ml/` y `frontend/` están versionados. El lockfile de pnpm 12 contiene dos documentos YAML por diseño; no deben separarse.

## Preparar, entrenar y evaluar

Desde la raíz `solar-vision/`:

```bash
cd ml
UV_PYTHON=3.12 uv sync --dev
PYTHONPATH=src uv run --no-sync python -m solar_ml.cli --archive ../dataset/2020-02-14_InfraredSolarModules.zip
```

La alternativa `--download` descarga el archivo del repositorio si GitHub está accesible. El pipeline verifica ZIP y metadatos y selecciona 400 imágenes por clase con semilla 42. La meta inicial de 1.000 se redujo para excluir globalmente parejas con dHash de distancia ≤ 2; el manifiesto final no contiene esas parejas. Se eliminaron dos duplicados exactos antes del muestreo. Se asignaron 280/60/60 imágenes por clase a entrenamiento/validación/prueba. Se comparó regresión logística con una MLP de 32 unidades; se eligió por F1 macro de validación y se evaluó una vez en prueba. El ZIP contiene 20.000 registros; las cuatro categorías elegidas tienen 10.000, 1.877, 1.288 y 1.499, en el orden indicado. [Detalles del pipeline](ml/README.md).

### Resultado medido

La regresión logística fue seleccionada (F1 macro de validación 0,539 frente a 0,427 de la MLP). En las 240 imágenes de prueba obtuvo accuracy **0,558**, precisión macro **0,564**, recall macro **0,558**, F1 macro **0,561** y AUC macro uno contra el resto **0,797**. Los valores por clase, matriz de confusión y curvas ROC están en `artifacts/metrics.json` y en el [informe PDF](report/informe.pdf). Ambos candidatos alcanzaron sus límites de iteraciones (300 y 60) sin convergencia declarada; la comparación no debe interpretarse como una búsqueda exhaustiva de hiperparámetros.

## Ejecutar API y frontend localmente

Después de entrenar, desde `solar-vision/`:

```bash
python3 backend/scripts/package_model.py
cd backend
UV_PYTHON=3.14 uv sync --locked --dev
uv run --no-sync uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

En otra terminal:

```bash
cd frontend
pnpm install --frozen-lockfile --ignore-scripts
pnpm dev
```

`--ignore-scripts` evita los scripts de instalación de las dependencias. `pnpm dev`, `pnpm typecheck`, `pnpm test` y `pnpm build` sí ejecutan código JavaScript; ejecútelos únicamente en un entorno autorizado, como el CI aislado. El build puede requerir que `esbuild` complete su instalación; en esta sesión no se aprobaron ni ejecutaron sus scripts.

Vite usa `/api` hacia `localhost:8000`. Compruebe `http://127.0.0.1:8000/api/health` y `/api/model/info` antes de enviar imágenes. La API recibe JSON como `{"pixels":[960 valores],"width":24,"height":40}`; no recibe el archivo original. `POST /api/predict` devuelve clase, confianza y probabilidades reales. Se pueden seleccionar hasta 20 imágenes por lote en la interfaz.

## Pruebas y PDF

```bash
cd ml
PYTHONPATH=src uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
cd ../backend
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync ruff check src tests
cd ../frontend
pnpm typecheck
pnpm test
pnpm build
cd ..
python3 report/build_report.py
typst compile --root . report/main.typ report/informe.pdf
```

`report/build_report.py` incorpora únicamente resultados de `artifacts/metrics.json` y exige las figuras generadas por el experimento. Sin ese archivo, el PDF declara la evaluación pendiente. Revise en el PDF los autores marcados como `[NOMBRE AUTOR ...]` antes de entregar. El guion de [video demostrativo](docs/video.md) utiliza resultados reales y dura menos de diez minutos.

## Cloudflare Workers

La configuración [Wrangler](backend/wrangler.jsonc) empaqueta la API Python y sirve `frontend/dist` mediante Workers Static Assets. Tras entrenar, empaquetar el modelo y construir el frontend:

```bash
cd backend
UV_PYTHON=3.14 uv sync --locked --dev
uv run --no-sync pywrangler deploy --dry-run
uv run --no-sync pywrangler dev
```

El dry run local empaquetó 9,666 MiB con pywrangler 1.17.9; esto comprueba el bundle, no la ejecución en Workers. En esta sesión `pywrangler dev` se detuvo al consultar `uv_interface_addresses` por restricciones del sandbox. En un entorno sin esa restricción, pruebe `/api/health`, `/api/model/info`, una predicción de una imagen real y un lote desde el navegador local de Workers. La publicación se prepara mediante [GitHub Actions](docs/ci-cloudflare.md): el workflow verifica ML, backend, frontend e informe y, en `main`, despliega y comprueba la URL pública. No se conoce el estado de los secretos de Cloudflare en GitHub ni se ha verificado la ejecución exitosa del workflow; **no hay URL pública verificada**. La viabilidad de FastAPI y Static Assets se apoya en [documentación oficial de Cloudflare](https://developers.cloudflare.com/workers/languages/python/packages/fastapi/). No guarde tokens ni credenciales en Git.

## Límites

La selección equilibrada de cuatro clases no representa las frecuencias reales de una planta. El dataset no provee identificadores de instalación; aunque el manifiesto final excluye parejas cercanas según dHash, la evaluación no garantiza independencia por sitio. El clasificador no localiza anomalías ni estima temperatura. La paridad de preprocesamiento JPEG y la inferencia en el runtime de Cloudflare requieren verificación adicional.
