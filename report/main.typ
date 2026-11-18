#import "@preview/charged-ieee:0.1.4": ieee
#import "results.typ": evaluation_summary, results_body, conclusion_status

#show: ieee.with(
  title: [Clasificación de anomalías térmicas en módulos fotovoltaicos mediante visión por computadora],
  authors: (
    (name: "[NOMBRE AUTOR 1]", department: [Departamento por completar], organization: [Institución por completar], location: [Ciudad por completar], email: "[CORREO AUTOR 1]"),
    (name: "[NOMBRE AUTOR 2]", department: [Departamento por completar], organization: [Institución por completar], location: [Ciudad por completar], email: "[CORREO AUTOR 2]"),
  ),
  abstract: [Se presenta el diseño de un clasificador de imágenes infrarrojas de módulos fotovoltaicos y una aplicación web para consultar sus predicciones. El trabajo utiliza cuatro categorías del conjunto InfraredSolarModules y compara regresión logística con una red neuronal densa compacta. La preparación elimina duplicados exactos antes de dividir los datos y reserva un conjunto de prueba para la evaluación final. La inferencia se diseñó para Cloudflare Workers mediante operaciones numéricas en Python y preprocesamiento determinista en el navegador. #evaluation_summary],
  index-terms: ("Visión por computadora", "Clasificación de imágenes", "Módulos fotovoltaicos", "Termografía", "Aprendizaje automático"),
  bibliography: bibliography("references.bib"),
)

= Introducción

La inspección termográfica permite observar diferencias de radiación en módulos fotovoltaicos que pueden acompañar anomalías de operación. La interpretación de esas imágenes requiere contexto y revisión especializada. Un clasificador puede ordenar imágenes para una primera revisión, pero sus etiquetas no constituyen un diagnóstico eléctrico definitivo. El conjunto InfraredSolarModules fue publicado para facilitar estudios de clasificación de anomalías en imágenes infrarrojas reales @millendorf2020.

Este proyecto implementa una ruta reproducible desde la preparación de datos hasta una aplicación web. Su objetivo académico es comparar dos clasificadores pequeños sobre cuatro categorías y exponer las probabilidades del modelo seleccionado. La aplicación recibe imágenes de módulos completas; no localiza defectos ni estima temperaturas absolutas.

= Dataset y preparación

InfraredSolarModules contiene 20 000 imágenes de 24 píxeles de ancho por 40 de alto, distribuidas en doce clases según sus autores @millendorf2020. Se eligieron *No-Anomaly*, *Cell*, *Cell-Multi* y *Diode*. El ZIP original contiene 10 000, 1 877, 1 288 y 1 499 registros en esas clases, respectivamente, según la validación del JSON de metadatos; las cantidades coinciden con las publicadas por el repositorio @raptormaps2020.

El pipeline valida el ZIP, su JSON de metadatos y cada imagen seleccionada. Aplica la semilla 42 y conserva 400 muestras por clase. La meta inicial de 1 000 se redujo porque no era viable seleccionar globalmente esa cantidad en las clases minoritarias manteniendo distancia dHash mayor que 2 entre imágenes. Antes de muestrear se eliminan duplicados exactos; la selección greedy excluye parejas próximas por dHash aun cuando sus etiquetas difieren. El manifiesto registra 1 600 imágenes sin parejas seleccionadas de distancia Hamming ≤ 2 y las divide en entrenamiento, validación y prueba con proporción 70 %, 15 % y 15 %. El repositorio no proporciona identificadores de instalación para garantizar independencia por sitio.

La entrada del modelo es un vector de 960 luminancias normalizadas. Se conservan las dimensiones originales. Para cada píxel, se redondea $0.299 R + 0.587 G + 0.114 B$ al entero más cercano y se divide entre 255. El navegador decodifica la imagen y aplica esa transformación; el Worker valida el tensor e infiere la clase. Las diferencias entre decodificadores JPEG del navegador y de Pillow requieren una comprobación de paridad con archivos reales.

= Metodología

La regresión logística proporciona una referencia simple. El segundo candidato es una red neuronal multicapa con 32 unidades ocultas, activación ReLU y salida probabilística de cuatro clases. Ambos usan exactamente la misma representación de imagen. La selección se basa en F1 macro de validación; el conjunto de prueba se reserva para una sola evaluación del modelo elegido. Se limitaron los ajustes a 300 iteraciones para la regresión y 60 para la MLP; ambas ejecuciones alcanzaron esos topes sin convergencia declarada. Esto condiciona la comparación y merece una revisión futura, pero no se modificaron hiperparámetros después de observar el conjunto de prueba.

La evaluación incluyó accuracy, precisión, recall y F1 macro, métricas por clase, matriz de confusión y curvas ROC uno contra el resto con AUC por clase y macro. El modelo elegido se exportó a JSON con pesos, sesgos, orden de clases y configuración de entrada. Una prueba comparó probabilidades de la biblioteca de entrenamiento con la implementación de inferencia exportable.

La arquitectura separa el entrenamiento en Python, una API FastAPI y la interfaz Vue 3. Cloudflare documenta FastAPI en Python Workers mediante `workers.asgi.entrypoint` @cloudflarefastapi y permite servir archivos estáticos en el mismo Worker @cloudflareassets. La API acepta un tensor de 24 píxeles de ancho por 40 de alto, valida dimensiones y valores finitos, y usa operaciones Python sin dependencias de inferencia pesadas. Esta elección evita asumir que frameworks de entrenamiento funcionen en el runtime de Workers.

#figure(
  grid(
    columns: (1fr,),
    row-gutter: 3pt,
    rect(width: 100%, fill: rgb("edf4f6"), stroke: rgb("87aeb7"), inset: 5pt)[*Navegador Vue* · decodificación y tensor],
    align(center)[↓],
    rect(width: 100%, fill: rgb("edf4f6"), stroke: rgb("87aeb7"), inset: 5pt)[*API FastAPI* · validación de 960 valores],
    align(center)[↓],
    rect(width: 100%, fill: rgb("edf4f6"), stroke: rgb("87aeb7"), inset: 5pt)[*Inferencia Python* · pesos exportados desde ML],
  ),
  caption: [Ruta de una predicción. Los archivos estáticos y la API se preparan para un mismo Worker.],
) <fig:architecture>

= Resultados y estado de verificación

#results_body

= Aplicación desarrollada

La @fig:architecture resume el recorrido de una solicitud. La interfaz permite seleccionar una o varias imágenes, conservar una vista previa con su resolución nativa, solicitar clasificaciones a la API y consultar probabilidades y un resumen del lote. El contrato HTTP incluye salud, información del modelo e inferencia. La respuesta se obtiene de los pesos exportados; si no existe un modelo entrenado empaquetado, la API devuelve un error de disponibilidad. Una prueba HTTP local de la aplicación ASGI con una imagen real de prueba devolvió una predicción, aunque clasificó un caso *Cell-Multi* como *Cell*. Es un error del modelo medido, no de transporte HTTP. El navegador no envía ni almacena imágenes en el servidor: transmite únicamente el tensor derivado.

El prototipo de infraestructura implementa la forma de integración descrita por la documentación oficial @cloudflarepython. Una prueba local en CPython midió la inferencia con pesos representativos, pero no equivale a una medición dentro de Workers. La validación con `pywrangler dev` y la URL pública deben comprobarse antes de presentar la aplicación como desplegada.

= Conclusiones

El código establece una arquitectura pequeña y reproducible para entrenar, comparar y publicar un clasificador térmico. Su utilidad potencial es ordenar imágenes para revisión humana, dentro de las cuatro clases estudiadas. #conclusion_status La resolución baja de las imágenes, la selección equilibrada para el experimento y la ausencia de identificadores de instalación limitan la generalización de cualquier resultado futuro.
