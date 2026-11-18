# Guion para video demostrativo (máximo 9:30)

**Condición previa:** el modelo ya fue entrenado con el ZIP original. Antes de grabar, compilar el frontend y verificar una inferencia real en `pywrangler dev` o en la URL pública. Si el despliegue público sigue pendiente, mostrar el Worker local y decirlo explícitamente. Usar las ocho imágenes reales de `ml/demo/`, extraídas de la partición de prueba, y conservar la atribución a Raptor Maps. No utilizar pesos representativos del prototipo como si fueran el modelo entrenado.

| Tiempo | Pantalla | Locución sugerida |
|---|---|---|
| 0:00–0:55 | Título y una imagen térmica real | «Este proyecto clasifica imágenes térmicas de módulos fotovoltaicos en cuatro categorías. Está pensado como apoyo a una revisión inicial, no como diagnóstico definitivo.» |
| 0:55–1:55 | Repositorio del dataset y `dataset_info.json` | «Usamos InfraredSolarModules, publicado por Raptor Maps. De las doce categorías seleccionamos No-Anomaly, Cell, Cell-Multi y Diode. El ZIP contiene 20.000 registros; las cuatro clases elegidas tienen 10.000, 1.877, 1.288 y 1.499 imágenes. Seleccionamos 400 por clase tras excluir imágenes demasiado próximas según dHash.» |
| 1:55–3:15 | `manifest.json`, código de entrenamiento y comparación | «El manifiesto distribuye 280, 60 y 60 imágenes por clase en entrenamiento, validación y prueba. No quedan parejas seleccionadas con distancia dHash menor o igual a dos. Comparamos regresión logística y una MLP de 32 neuronas sobre 960 valores de luminancia. Elegimos la regresión por F1 macro en validación: 0,539 frente a 0,427.» |
| 3:15–4:45 | `metrics.json`, matriz de confusión y ROC | «En 240 imágenes de prueba, la regresión logística obtuvo accuracy 0,558, precisión macro 0,564, recall macro 0,558, F1 macro 0,561 y AUC macro 0,797. La mayor confusión fue Cell-Multi predicha como Cell, en 21 casos. El AUC fue 0,784, 0,775, 0,719 y 0,912, respectivamente, para las cuatro clases.» |
| 4:45–5:25 | Arquitectura de README, `/api/health` y `/api/model/info` | «El navegador convierte la imagen de 24 píxeles de ancho por 40 de alto a grises sin ampliar su resolución; envía un tensor a la API FastAPI diseñada para Cloudflare Workers. El backend ejecuta exactamente los pesos exportados y devuelve probabilidades.» |
| 5:25–7:45 | App real: una imagen por clase, preview y probabilidades | «Cargo imágenes reales del conjunto de prueba. Para cada una se muestran clase, confianza y distribución de probabilidades. Una predicción incorrecta también es informativa; la revisamos frente a la etiqueta original.» |
| 7:45–8:40 | Carga múltiple de 4–8 imágenes y resumen | «La selección múltiple clasifica cada archivo y resume cuántas predicciones corresponden a cada categoría. Las imágenes no se almacenan en el backend.» |
| 8:40–9:25 | PDF y conclusión | «El informe reúne la evaluación y sus límites: baja resolución, cuatro clases elegidas y falta de identificadores de instalación. El sistema sirve para priorizar inspección visual, no para tomar decisiones eléctricas por sí solo.» |

## Preparación de la grabación

1. Sustituir marcadores de autor en `report/main.typ`; regenerar y revisar `report/informe.pdf`.
2. Usar las dos imágenes reales por clase de `ml/demo/` y consultar `ml/demo/provenance.json` para sus etiquetas; no mezclar imágenes de entrenamiento en la demostración de evaluación.
3. Dejar abiertos `metrics.json`, figuras, aplicación y endpoint. Verificar una petición real justo antes de grabar.
4. Ensayar los tiempos. Mostrar la URL pública solo si fue probada; en otro caso, indicar que se usa `pywrangler dev`.
5. Cerrar el video antes de 9:30 para dejar margen al límite de diez minutos.
