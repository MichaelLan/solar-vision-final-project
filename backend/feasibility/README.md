# Validación de Cloudflare Workers

Fecha: 16 de noviembre de 2026. Este directorio contiene un prototipo aislado; sus pesos son **representativos y no entrenados**. La aplicación de producción solo utiliza el modelo exportado desde `ml/`.

## Evidencia y decisión

- La [documentación oficial de Python Workers](https://developers.cloudflare.com/workers/languages/python/) exige `python_workers` y ejecuta el entorno local mediante `uv run pywrangler dev`.
- La [guía oficial de FastAPI](https://developers.cloudflare.com/workers/languages/python/packages/fastapi/) muestra `workers.asgi.entrypoint(app)` y la integración de Static Assets.
- Los [límites oficiales](https://developers.cloudflare.com/workers/platform/limits/) indican 128 MB por isolate y 10 ms de CPU en Workers Free. El modelo final es una regresión logística multiclase de una capa, con inferencia en Python puro y pesos incluidos como módulo Python en el Worker. No se presupone compatibilidad de TensorFlow, PyTorch ni ONNX Runtime.
- Cloudflare [usa Python 3.14 por defecto](https://developers.cloudflare.com/workers/configuration/compatibility-flags/) desde la fecha de compatibilidad 2026-09-08. El backend final exige Python 3.14; el prototipo aislado se creó inicialmente con 3.13.
- [Static Assets](https://developers.cloudflare.com/workers/static-assets/) permite servir la compilación de Vite en el mismo Worker. La API procesa tensores de 24 píxeles de ancho × 40 de alto, orientación verificada en el ZIP original; el navegador decodifica los JPEG y aplica la fórmula de luminancia documentada en el proyecto.

## Pruebas ejecutadas

- La descarga por PyPI sigue fallando por DNS. Con el caché de `uv` preexistente y Python 3.14, `uv sync --offline` instaló FastAPI 0.162.2, workers-py 1.17.9 y sus dependencias; generó `uv.lock`.
- `pywrangler` exigió `uv >= 0.13.4`; el ejecutable local era 0.13.0. Se ejecutó con `uv 0.13.7` desde el caché, sin descargar paquetes. Wrangler 4.158.0 se resolvió desde el caché local de npm.
- `pywrangler deploy --dry-run --outdir /tmp/solar-worker-bundle` pasó con el modelo **entrenado real**: 436 módulos, 9.665,85 KiB de módulos y 9.666,39 KiB de carga total (2.415,85 KiB comprimidos). El módulo de pesos ocupa 178,18 KiB. Se usó un `index.html` temporal para satisfacer el directorio Static Assets; se eliminó tras la prueba. Esta prueba comprueba el empaquetado, no la inferencia en Workers ni el frontend final.
- `pywrangler dev` llegó a Wrangler, que detectó el binding `ASSETS`, pero se detuvo con `uv_interface_addresses returned Unknown system error 1`, incluso con `--ip 127.0.0.1`. El sandbox no permitió arrancar el servidor local; por ello no se midió latencia ni memoria en el runtime.
- La ruta ASGI de FastAPI sí se comprobó con el modelo entrenado y un JPEG real: `/api/predict` respondió 200 y coincidió con las cuatro probabilidades de referencia de ML. Ocho pruebas backend pasaron.
- `python3.13 benchmark.py`: artefacto representativo de 39.932 bytes; mediana 0,834 ms, percentil 95 1,319 ms en 100 inferencias; pico asignado por `tracemalloc` 5.080 bytes durante el bucle. Esta medida corresponde **solo a CPython local**, no al runtime de Cloudflare.

No fue posible completar `pywrangler dev`, medir memoria del isolate, comprobar HTTP en el runtime ni desplegar públicamente con el acceso de red actual. Con red y entorno local normal, ejecutar desde `backend/`:

```bash
uv sync
uv run pywrangler dev
curl http://127.0.0.1:8787/api/health
```

En otra terminal, enviar a `/api/predict` un JSON con `pixels` de longitud 960 y valores entre 0 y 1, `width: 24`, `height: 40`. Los pesos del prototipo no deben citarse como evaluación del proyecto.
