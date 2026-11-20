#import "@preview/charged-ieee:0.1.4": ieee
#import "results.typ": v, validation_rows, class_rows, comparison_rows

// The template hard-codes English labels; translate them for a Spanish paper.
#show "Abstract": "Resumen"
#show "Index Terms": "Palabras clave"
#show "TABLE": "TABLA"
#set text(lang: "es", region: "co")

#show: ieee.with(
  title: [Clasificación de anomalías térmicas en módulos fotovoltaicos con una red convolucional compacta],
  authors: (
    (
      name: "Michael Lan Alvarez",
      department: [Maestría en Inteligencia Artificial],
      organization: [Universidad de La Salle],
      location: [Bogotá, Colombia],
    ),
  ),
  abstract: [
    La termografía aérea permite revisar plantas fotovoltaicas extensas, pero produce miles de imágenes que alguien debe interpretar. Este trabajo aplica visión por computadora para clasificar imágenes infrarrojas de módulos individuales en cuatro patrones térmicos (No-Anomaly, Cell, Cell-Multi y Diode), a partir del conjunto público InfraredSolarModules. Se preparó una muestra equilibrada de #v.all_total imágenes sin copias repetidas entre entrenamiento y prueba, y se compararon una regresión logística sobre los píxeles y dos redes convolucionales pequeñas, con reglas de selección fijadas antes de entrenar. La red elegida, de #v.params_w16 parámetros, obtuvo en un conjunto de prueba nuevo, evaluado una sola vez, una exactitud de #v.acc, un F1 macro de #v.f1 y un AUC macro de #v.auc, frente a un F1 macro de #v.base_f1 de la regresión logística. Como detector de anomalía frente a módulo normal alcanzó una sensibilidad de #v.sens y una especificidad de #v.spec. La confusión principal ocurre entre anomalías de una y de varias celdas. El modelo se integró en una aplicación web pública que clasifica imágenes individuales o lotes.
  ],
  index-terms: (
    "Visión por computadora",
    "redes neuronales convolucionales",
    "termografía infrarroja",
    "módulos fotovoltaicos",
    "clasificación de imágenes",
  ),
  bibliography: bibliography("references.bib", title: [Referencias]),
)

#show figure.where(kind: table): set figure(supplement: [Tabla])
#show link: set text(hyphenate: false)

= Introducción

La operación y el mantenimiento de plantas fotovoltaicas requieren detectar a tiempo los módulos que generan menos energía de la esperada. Muchas de esas fallas se manifiestan como calentamiento localizado: una celda dañada o sombreada disipa potencia en lugar de producirla, y un diodo de bypass activado hace que una fracción del módulo opere a mayor temperatura. La termografía infrarroja captada desde drones permite observar estos patrones en plantas de gran tamaño, pero la revisión manual de miles de imágenes es lenta y su interpretación depende de las condiciones de captura. Por ello, automatizar la detección y clasificación de fallas es un tema activo, en el que la exactitud y la generalización a más clases y plantas siguen siendo problemas abiertos @oliveira2022.

Este proyecto aborda una parte acotada del problema: clasificar la imagen térmica de un módulo en una de cuatro categorías para ayudar a un técnico a decidir qué revisar primero. Para ello se preparó un subconjunto de datos limpio, se compararon un modelo lineal y una red convolucional, se evaluó el modelo elegido en imágenes no vistas y se integró en una aplicación web. El trabajo no localiza el defecto dentro del módulo, no estima temperaturas y no reemplaza el diagnóstico eléctrico. Las secciones siguientes describen los datos, el método, los resultados y la aplicación.

= Datos y preparación

== Conjunto de datos
Se utilizó InfraredSolarModules, publicado por Raptor Maps @millendorf2020 @raptormaps2020. Contiene #v.zip_total imágenes infrarrojas de módulos individuales, en escala de grises y de #box[24 × 40] píxeles, repartidas en #v.zip_classes clases: once tipos de anomalía y la clase sin anomalía, que reúne el #v.prev_all~% del total. El artículo que presenta el conjunto no publica resultados de clasificación de referencia @millendorf2020, por lo que no hay cifras directamente comparables con las de este trabajo.

Se eligieron cuatro clases que describen el estado térmico del propio módulo: *No-Anomaly* (#v.n_noanomaly imágenes), sin anomalía identificada; *Cell* (#v.n_cell), un punto caliente en una celda; *Cell-Multi* (#v.n_cellmulti), puntos calientes en varias celdas; y *Diode* (#v.n_diode), un diodo de bypass activado, que suele calentar un tercio del módulo @raptormaps2020. La @fig:examples muestra ejemplos de cada una. Vegetation, Shadowing y Soiling se excluyeron porque responden a causas externas o pasajeras (plantas, sombras, suciedad); Hot-Spot y Hot-Spot-Multi corresponden a otra tecnología de módulo (película delgada); Diode-Multi tiene solo #v.n_diode_multi imágenes; y Cracking y Offline-Module quedaron fuera para mantener un problema acotado.

#figure(
  image("figures/class_examples.png", width: 100%),
  placement: auto,
  caption: [Seis imágenes de entrenamiento por clase, elegidas al azar. Cada imagen se escala a su propio rango y se muestra con un mapa de color térmico (oscuro: frío; claro: caliente).],
) <fig:examples>

== Eliminación de duplicados y partición
El conjunto contiene imágenes repetidas o casi idénticas. Si una copia queda en entrenamiento y otra en prueba, el modelo se evalúa con imágenes que ya conoce y el resultado se infla. Por eso, además de quitar #v.exact_dups duplicados exactos, se comparó cada imagen candidata con las ya seleccionadas de cualquier clase y partición, mediante la raíz de la diferencia cuadrática media (RMS) de sus píxeles en escala $[0, 1]$. Una candidata se descartó si esa diferencia era de #v.rms_threshold o menos, es decir, unos #v.rms_threshold_gray niveles de gris de 255 en promedio. Este umbral está muy por debajo de la diferencia habitual entre dos imágenes cualesquiera, de modo que solo elimina copias casi exactas. En total se descartaron #v.rejected_total imágenes.

Las imágenes de cada clase se recorrieron en orden aleatorio (semilla #v.seed) y se asignaron hasta completar #v.test_pc para prueba, #v.val_pc para validación y #v.train_pc para entrenamiento (@tab:splits). La prueba se formó solo con imágenes que no se habían consultado antes, porque en una etapa anterior del proyecto se había reutilizado el conjunto de prueba; esa evaluación previa (exactitud de #v.hist_acc con regresión logística, en otra partición) no es comparable y se descartó. La partición equilibrada permite comparar las clases con el mismo peso, aunque no refleja su frecuencia real: entre las cuatro clases, No-Anomaly representa el #v.prev_four~% del conjunto original. Además, el conjunto no identifica la planta ni el vuelo de cada imagen, así que no puede garantizarse que entrenamiento y prueba provengan de instalaciones distintas.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 2.5pt),
    table.hline(),
    table.header([Clase], [Original], [Entrenam.], [Validación], [Prueba]),
    table.hline(stroke: 0.5pt),
    [No-Anomaly], [#v.n_noanomaly], [#v.train_pc], [#v.val_pc], [#v.test_pc],
    [Cell], [#v.n_cell], [#v.train_pc], [#v.val_pc], [#v.test_pc],
    [Cell-Multi], [#v.n_cellmulti], [#v.train_pc], [#v.val_pc], [#v.test_pc],
    [Diode], [#v.n_diode], [#v.train_pc], [#v.val_pc], [#v.test_pc],
    table.hline(stroke: 0.5pt),
    [Total], [#v.n_four], [#v.train_total], [#v.val_total], [#v.test_total],
    table.hline(),
  ),
  caption: [Imágenes por clase en el conjunto original y en la partición usada],
) <tab:splits>

== Preprocesamiento
Las imágenes se usan a su resolución original, sin redimensionar, y sus 960 píxeles se escalan a $[0, 1]$. Para la red convolucional, cada imagen se estandariza además con su propia media y desviación estándar. Como los niveles de gris no son temperaturas calibradas, este paso elimina las diferencias de brillo general entre capturas y conserva el contraste relativo, que es lo que delata un punto caliente.

= Metodología

Con los datos preparados, se compararon dos tipos de modelo: uno lineal que trata cada píxel por separado y uno convolucional que busca patrones locales.

== Modelos
La *regresión logística* sobre los 960 píxeles (#v.params_lr parámetros) sirve como línea base. Asigna un peso fijo a cada posición de la imagen, por lo que no puede reconocer el mismo punto caliente si aparece en otro lugar del módulo. Se usó con regularización L2 ($C = 1$), sobre los píxeles en $[0, 1]$ y sin aumento de datos.

Una *red convolucional* (CNN), en cambio, aplica los mismos filtros pequeños en todas las posiciones de la imagen y combina los patrones detectados en capas sucesivas @lecun1998. Esto se ajusta bien al problema: un punto caliente es un patrón local que puede estar en cualquier celda, y lo que distingue a las clases es la forma y el número de regiones calientes. Dado el tamaño de las imágenes y de la muestra, se diseñó una red pequeña con tres capas convolucionales de filtros #box($3 times 3$) con ReLU y relleno 1, sin normalización por lotes. Tras cada una de las dos primeras, un _max-pooling_ #box($2 times 2$) reduce el mapa a la mitad: de #box($40 times 24$) a #box($20 times 12$) y luego a #box($10 times 6$). Así, cada unidad de la última capa observa una zona de #box($18 times 18$) píxeles de la imagen, que abarca varias celdas. Al final, un promedio global resume cada mapa de activación sin importar dónde apareció el patrón y, tras un _dropout_, una capa densa con _softmax_ produce la probabilidad de cada clase. Se probaron dos anchos: w8, con 8, 16 y 32 filtros por capa (#v.params_w8 parámetros), y w16, con el doble (#v.params_w16).

Conviene notar que la comparación no aísla el efecto de la convolución: la CNN recibe además la estandarización por imagen y el aumento de datos, que la regresión no usa. Por ello, la mejora observada se atribuye al conjunto de arquitectura y preprocesamiento.

== Entrenamiento
La @tab:hparams resume la configuración, que se fijó de antemano y se aplicó igual a ambos anchos, sin búsqueda de hiperparámetros. Como aumento de datos se usaron volteos horizontales y verticales, porque la clase de una anomalía no depende de la orientación del módulo en la imagen. La regresión logística se entrenó con scikit-learn #v.sklearn @pedregosa2011 y las CNN con PyTorch #v.torch @paszke2019.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: none,
    inset: (x: 4pt, y: 2.5pt),
    table.hline(),
    table.header([Parámetro], [Valor]),
    table.hline(stroke: 0.5pt),
    [Entrada], [$1 times 40 times 24$, estandarizada por imagen],
    [Aumento], [Volteo horizontal y vertical, probabilidad 0,5 cada uno],
    [Optimizador], [AdamW @loshchilov2019, decaimiento de pesos #v.wd#[;] tasa de aprendizaje con ciclo único (OneCycle), máximo #v.lr],
    [Pérdida], [Entropía cruzada, suavizado de etiquetas #v.smoothing],
    [Regularización], [_Dropout_ #v.dropout antes de la capa densa],
    [Lote y épocas], [#v.batch imágenes; #v.epochs épocas, sin parada temprana],
    [Pesos finales], [Los de la última época],
    [Semillas], [#v.seeds],
    table.hline(),
  ),
  caption: [Configuración de entrenamiento de las CNN],
) <tab:hparams>

== Selección del modelo
Las reglas de selección se registraron antes de entrenar. Cada CNN se entrenó con #v.n_seeds semillas (#v.seeds) para ver cuánto varía el resultado entre ejecuciones. Se elegiría el ancho con mayor F1 macro medio en validación, salvo que w8 quedara a #v.tie_margin o menos de w16, en cuyo caso se preferiría w8 por ser más simple. El modelo final sería la ejecución con semilla #v.seed, fijada de antemano en lugar de escoger la mejor. La prueba se usaría una sola vez, para el modelo elegido y para la línea base. La @fig:training muestra la salida del programa de entrenamiento al ejecutarse este procedimiento.

#figure(
  image("figures/training_console.png", width: 80%),
  caption: [Salida de consola de una reproducción del entrenamiento (20 de noviembre de 2026), cuyos resultados coinciden con los informados: pérdida de entrenamiento cada 10 épocas, F1 macro y exactitud de validación por semilla, resumen por modelo y selección. Se omiten bloques intermedios («...»).],
  placement: top,
  scope: "parent",
) <fig:training>

== Métricas
Se calculan por clase la precisión (fracción de las predicciones de una clase que son correctas), el _recall_ (fracción de imágenes de la clase que el modelo reconoce) y el F1, que combina ambos; además, la exactitud global y los promedios macro, que dan el mismo peso a cada clase. La matriz de confusión muestra qué clases se confunden entre sí. Para las curvas ROC, cada clase se compara con las otras tres usando la probabilidad que el modelo le asigna. El área bajo la curva (AUC) indica con qué frecuencia una imagen de esa clase recibe más probabilidad que una de otra clase @fawcett2006; el AUC macro promedia las cuatro. A diferencia de la exactitud, el AUC no depende de elegir la clase más probable, por lo que puede ser alto aunque haya confusiones apreciables.

Como la prueba tiene solo #v.test_total imágenes, cada métrica se acompaña de un intervalo de confianza del 95 % obtenido por _bootstrap_: se repitió #v.resamples veces el cálculo sobre muestras de la prueba tomadas con reemplazo @efron1993. El intervalo refleja la incertidumbre por evaluar con solo #v.test_total imágenes, no la variación entre entrenamientos. Por último, para el uso práctico, se agruparon las tres anomalías en una sola clase y se midió la capacidad del modelo para separar anomalía de módulo normal (sensibilidad y especificidad).

= Resultados

== Selección en validación
La @tab:validation resume la validación. Ambas CNN superan con amplitud a la regresión logística. La w16 supera a la w8 por #v.w16_w8_gap en F1 macro medio, más que el margen de #v.tie_margin y que la variación entre semillas, por lo que se seleccionó *CNN w16*. Su ejecución con semilla #v.seed obtuvo un F1 macro de validación de #v.w16_seed_f1, y todas las cifras de prueba corresponden a ella.

#figure(
  table(
    columns: 5,
    align: (left, right, center, center, center),
    stroke: none,
    inset: (x: 3pt, y: 2.5pt),
    table.hline(),
    table.header([Modelo], [Parám.], [Semillas], [F1 macro], [AUC macro]),
    table.hline(stroke: 0.5pt),
    ..validation_rows,
    table.hline(),
  ),
  caption: [Validación (#v.val_total imágenes): media ± desviación estándar entre semillas],
) <tab:validation>

== Evaluación en prueba
En el conjunto de prueba nuevo, CNN w16 obtuvo una exactitud de #v.acc (IC 95 %: #v.acc_ci), un F1 macro de #v.f1 #v.f1_ci y un AUC macro de #v.auc #v.auc_ci, con precisión y _recall_ macro de #v.prec y #v.rec. Estos valores son cercanos a los de validación, lo que sugiere que la selección no se ajustó en exceso a esa partición. La regresión logística, evaluada con las mismas imágenes, obtuvo una exactitud de #v.base_acc y un F1 macro de #v.base_f1 #v.base_f1_ci (@tab:test). Los intervalos de confianza de ambos modelos no se solapan, lo que respalda la diferencia de desempeño observada. Sin embargo, no se realizó una estimación específica de la incertidumbre de esa diferencia.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    inset: (x: 3pt, y: 2.5pt),
    table.hline(),
    table.header([Modelo], [Exactitud], [F1 macro], [AUC macro], [Sensib.], [Especif.]),
    table.hline(stroke: 0.5pt),
    ..comparison_rows,
    table.hline(),
  ),
  caption: [Prueba (#v.test_total imágenes). Sensibilidad y especificidad al separar anomalía de módulo normal],
) <tab:test>

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    inset: (x: 3pt, y: 2.5pt),
    table.hline(),
    table.header([Clase], [Precisión], [_Recall_], [F1], [AUC], [F1 RL]),
    table.hline(stroke: 0.5pt),
    ..class_rows,
    table.hline(),
  ),
  caption: [Métricas por clase de CNN w16 en prueba (#v.test_pc imágenes por clase); F1 RL: regresión logística],
) <tab:classes>

#figure(
  image("figures/confusion_matrix.png", width: 100%),
  caption: [Matrices de confusión en prueba: filas, clase real; columnas, clase predicha. Cada celda indica el número de imágenes y el porcentaje de su fila.],
  placement: top,
  scope: "parent",
) <fig:confusion>

El rendimiento varía mucho según la clase (@tab:classes y @fig:confusion). *Diode* es la más fácil (F1 #v.diode_f1, _recall_ #v.diode_recall): su patrón es una franja caliente que ocupa una parte grande del módulo y se distingue incluso a baja resolución. *No-Anomaly* tiene un _recall_ alto (#v.normal_recall) y una precisión menor (#v.normal_prec), porque también recibe algunas anomalías que el modelo no detecta. Las clases difíciles son *Cell* (F1 #v.cell_f1) y *Cell-Multi* (F1 #v.multi_f1). Entre ellas se producen #v.cell_confusion de los #v.errors_total errores (#v.cell_confusion_share~%): #v.cell_to_multi imágenes de Cell se predijeron como Cell-Multi y #v.multi_to_cell en sentido contrario. El siguiente error en frecuencia es una anomalía pequeña clasificada como normal (#v.cell_to_normal de Cell y #v.multi_to_normal de Cell-Multi).

Estas confusiones se explican en buena parte por la resolución. Cell y Cell-Multi se diferencian por el número de celdas calientes, y la baja resolución de las imágenes dificulta distinguir anomalías tan similares. En particular, los puntos calientes de varias celdas cercanas pueden aparecer como una sola región, mientras que las anomalías pequeñas pueden confundirse con las variaciones normales de la imagen. La @fig:errors lo ilustra con los errores en que el modelo estaba más seguro. Las imágenes de Cell tomadas por Cell-Multi muestran una mancha grande o alargada; las de Cell-Multi tomadas por Cell, una sola región caliente pequeña en el borde; y las anomalías tomadas por normales tienen poco contraste. Algunas de estas imágenes son ambiguas también para un observador humano, por lo que parte del error puede deberse al etiquetado.

#figure(
  image("figures/error_examples.png", width: 100%),
  caption: [Errores de prueba con mayor probabilidad asignada, por tipo: clase real → predicha y probabilidad de la clase predicha (NoA: No-Anomaly; #box[C-M]: Cell-Multi).],
) <fig:errors>

Las curvas ROC (@fig:roc) confirman este orden. Diode (AUC #v.diode_auc) y No-Anomaly (#v.normal_auc) se separan bien del resto, mientras que Cell (#v.cell_auc) y Cell-Multi (#v.multi_auc) tienen curvas más bajas; la matriz de confusión muestra que se confunden sobre todo entre sí. La CNN supera a la regresión logística en el AUC de las cuatro clases. La mayor ganancia en F1 está en No-Anomaly, que sube de #v.base_normal_f1 a #v.normal_f1, porque la regresión confundía muchas imágenes normales con anomalías de celda; Cell y Cell-Multi pasan de #v.base_cell_f1 y #v.base_multi_f1 a #v.cell_f1 y #v.multi_f1. Esta mejora es coherente con la capacidad de la convolución para reconocer un punto caliente en cualquier posición (sin separar el efecto del preprocesamiento).

#figure(
  image("figures/roc_curves.png", width: 92%),
  placement: auto,
  caption: [Curvas ROC de cada clase frente a las demás, en prueba. Líneas continuas: CNN w16 (AUC en la leyenda); punteadas: regresión logística, mismo color por clase.],
) <fig:roc>

== Utilidad y límites
Para priorizar inspecciones, lo más relevante es si el modelo separa anomalía de módulo normal. La CNN marcó como anómalas #v.detected de las #v.positives imágenes con anomalía (sensibilidad #v.sens) y dio #v.false_alarms falsas alarmas en #v.negatives imágenes normales (especificidad #v.spec, frente a #v.base_spec de la línea base). Como dejó pasar #v.missed anomalías, sirve para ordenar la revisión, no para descartar módulos sin mirarlos. Estas cifras se midieron con clases equilibradas; en el conjunto original, donde los módulos normales son el #v.prev_four~% de las cuatro clases, y con la misma sensibilidad y especificidad, cabría esperar que alrededor del #v.ppv_field~% de las alarmas correspondiera a anomalías reales.

Los resultados tienen además límites claros. Proceden de un único conjunto de datos, con un mismo origen y cámara y sin identificación de planta, por lo que no miden la generalización a otras instalaciones. La resolución de #box[24 × 40] píxeles limita la distinción entre una y varias celdas. Solo se consideran cuatro de las doce clases, de modo que una imagen de otra categoría, como sombra o suciedad, se asignará a una de las cuatro; los autores del conjunto advierten, además, que sus proporciones no reproducen las de campo @millendorf2020. Por último, las probabilidades no están calibradas y no deben leerse como una medida de confianza diagnóstica.

= Aplicación desarrollada

Para llevar el clasificador a un uso práctico, se desarrolló una aplicación web pública, disponible en #box(link("https://solar-vision.michaellan-eng.workers.dev/")[solar-vision.michaellan-eng.workers.dev]). El usuario carga una imagen térmica de #box[24 × 40] píxeles, en el formato de escala de grises del conjunto, o un lote de hasta 20, y la aplicación muestra para cada una la clase estimada y la probabilidad de cada clase, además de un resumen del lote. La @fig:app muestra el resultado para una imagen de prueba. Las imágenes no se almacenan.

#figure(
  image("figures/app_result.png", width: 82%),
  caption: [Captura de la aplicación pública: vista previa de una imagen de prueba de Cell-Multi (izquierda), clasificada correctamente como «Anomalía en varias celdas», y probabilidad de cada clase (derecha).],
  placement: top,
  scope: "parent",
) <fig:app>

#figure(
  {
    set text(size: 7pt, hyphenate: false)
    let node(title, body) = box(
      stroke: 0.6pt,
      radius: 2pt,
      inset: 4pt,
      width: 100%,
      align(center)[*#title* \ #body],
    )
    let arrow(top, bottom) = align(center + horizon)[#top \ #sym.arrow.r.long \ #bottom]
    grid(
      columns: (1fr, auto, 1fr, auto, 1fr),
      column-gutter: 3pt,
      align: horizon,
      node[Navegador][Vue 3 \ escala de grises],
      arrow[960 valores][`/api/predict`],
      node[Worker][FastAPI \ estandariza],
      arrow[tensor][],
      node[CNN][numpy \ pesos JSON],
    )
  },
  kind: image,
  caption: [Arquitectura de inferencia. La respuesta devuelve la clase y las cuatro probabilidades.],
) <fig:arch>


La @fig:arch resume su funcionamiento. El navegador convierte la imagen en los mismos 960 valores de gris usados en el entrenamiento y los envía a una API escrita en FastAPI, que se ejecuta en un Worker de Python de Cloudflare junto con la interfaz @cloudflarepython @cloudflarefastapi @cloudflareassets. El servidor aplica la misma estandarización y ejecuta la red. Como PyTorch no está disponible en ese entorno, la red se reimplementó con numpy a partir de los pesos entrenados; esta versión predice la misma clase que el modelo original en las #v.test_total imágenes de prueba, y la aplicación pública sirve el modelo evaluado en este informe: la huella SHA-256 de los pesos que informa la API coincide con la del modelo exportado.

= Conclusiones

Este trabajo mostró que una red convolucional pequeña puede clasificar imágenes térmicas de baja resolución de módulos fotovoltaicos en cuatro patrones con un rendimiento útil. Con #v.params_w16 parámetros, la red alcanzó en una prueba nueva un F1 macro de #v.f1 y un AUC macro de #v.auc, claramente por encima de la regresión logística (#v.base_f1 de F1 macro), con intervalos de confianza que no se solapan. Como la red usó además estandarización por imagen y aumento de datos, esta mejora corresponde al conjunto de arquitectura y preprocesamiento. La eliminación de duplicados entre particiones y el uso único de la prueba, con reglas fijadas antes de entrenar, reducen el riesgo de sobrestimar el rendimiento dentro de este conjunto de datos.

El modelo reconoce de forma fiable el patrón de diodo y la ausencia de anomalía, y como detector de anomalías alcanza una sensibilidad de #v.sens. En cambio, distinguir una celda caliente de varias sigue siendo difícil a #box[24 × 40] píxeles. Como trabajo futuro se propone evaluar el modelo con imágenes de otras plantas, usar imágenes de mayor resolución o con temperatura radiométrica, calibrar las probabilidades e incluir las demás clases del conjunto, en especial sombras y suciedad, que hoy se asignarían a alguna de las cuatro categorías.
