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
    La termografía aérea permite revisar plantas fotovoltaicas extensas, pero produce miles de imágenes que deben clasificarse. Este trabajo entrena y evalúa un clasificador de cuatro patrones térmicos (No-Anomaly, Cell, Cell-Multi y Diode) sobre el conjunto público InfraredSolarModules y lo integra en una aplicación web. Se formó una partición equilibrada de #v.all_total imágenes sin duplicados ni casi duplicados entre particiones, con un conjunto de prueba que se evaluó una sola vez. Con un protocolo fijado antes del entrenamiento se compararon una regresión logística sobre píxeles y dos redes convolucionales compactas. La red seleccionada, de #v.params_w16 parámetros, obtuvo en prueba una exactitud de #v.acc, un F1 macro de #v.f1 y un AUC macro uno contra el resto de #v.auc, frente a #v.base_f1 de F1 macro de la regresión logística. Como detector binario de anomalía alcanzó una sensibilidad de #v.sens y una especificidad de #v.spec. La confusión principal ocurre entre anomalías de una y de varias celdas. El modelo se sirve mediante una API en un Worker de Cloudflare y una interfaz que clasifica imágenes individuales o lotes.
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

La operación y el mantenimiento de plantas fotovoltaicas requieren detectar a tiempo los módulos que generan menos energía de la esperada. Muchas de esas fallas se manifiestan como calentamiento localizado: una celda dañada o sombreada disipa potencia en lugar de producirla, y un diodo de bypass activado hace que una fracción del módulo opere a mayor temperatura. La termografía infrarroja captada desde drones permite observar estos patrones en plantas de gran tamaño, pero la revisión manual de miles de imágenes es lenta y su interpretación depende de las condiciones de captura @oliveira2022. Por ello, la automatización de la detección y clasificación de fallas es un tema activo, en el que la exactitud, la robustez y la generalización a más clases y plantas siguen siendo problemas abiertos @oliveira2022.

Este proyecto aborda una parte acotada del problema: clasificar imágenes térmicas de módulos individuales en cuatro categorías para priorizar la revisión de un técnico. Los objetivos son 1) preparar un subconjunto de datos sin fugas entre particiones; 2) seleccionar un modelo mediante un protocolo definido antes de entrenar; 3) evaluarlo una sola vez en un conjunto de prueba nuevo, con intervalos de confianza; y 4) integrarlo en una aplicación web que acepte imágenes individuales o lotes. El trabajo no localiza el defecto dentro del módulo, no estima temperaturas absolutas y no reemplaza el diagnóstico eléctrico.

= Datos y preparación

== Conjunto de datos
Se utilizó InfraredSolarModules, publicado por Raptor Maps @millendorf2020 @raptormaps2020. Contiene #v.zip_total imágenes infrarrojas de módulos individuales, recortadas a #box[24 × 40] píxeles en escala de grises, con #v.zip_classes clases: once tipos de anomalía y la clase sin anomalía, que representa el #v.prev_all~% del total. El artículo que presenta el conjunto describe las clases y sus proporciones, pero no publica resultados de clasificación de referencia, que propone como trabajo futuro @millendorf2020; por ello no hay cifras directamente comparables con las de este trabajo.

Se eligieron cuatro clases que describen estados térmicos del propio módulo cristalino: *No-Anomaly* (#v.n_noanomaly imágenes), sin anomalía identificada; *Cell* (#v.n_cell), un punto caliente en una celda; *Cell-Multi* (#v.n_cellmulti), puntos calientes en varias celdas; y *Diode* (#v.n_diode), el patrón de un diodo de bypass activado, que suele abarcar un tercio del módulo @raptormaps2020. Vegetation (#v.n_vegetation), Shadowing (#v.n_shadowing) y Soiling (#v.n_soiling) se excluyeron porque responden a causas externas o transitorias (plantas, sombras, suciedad) más que al estado eléctrico del módulo; Hot-Spot y Hot-Spot-Multi corresponden a módulos de película delgada; y Diode-Multi tiene solo #v.n_diode_multi imágenes. Cracking (#v.n_cracking) y Offline-Module (#v.n_offline) quedaron fuera para mantener un problema acotado. La @fig:examples muestra ejemplos de entrenamiento; todas las imágenes de las cuatro clases son JPEG de #box[24 × 40] píxeles y un canal.

#figure(
  image("figures/class_examples.png", width: 100%),
  placement: auto,
  caption: [Seis imágenes de entrenamiento por clase, elegidas al azar. Cada imagen se escala a su propio rango y se muestra con un mapa de color térmico (oscuro: frío; claro: caliente).],
) <fig:examples>

== Representación de la entrada
Cada imagen se convierte a escala de grises y se escala a $[0, 1]$, lo que produce 960 valores. No se redimensiona ni se interpola: la resolución de entrada es la original. Este vector alimenta directamente a la regresión logística. Para las redes convolucionales se reorganiza como un tensor de $1 times 40 times 24$ y se estandariza por imagen, $x' = (x - mu) slash (sigma + 10^(-6))$, donde $mu$ y $sigma$ son la media y la desviación estándar de los píxeles de esa imagen. Como los niveles de gris de estas imágenes no son temperaturas calibradas, la estandarización elimina diferencias globales de brillo entre capturas y conserva el contraste relativo, que es lo que distingue un punto caliente.

== Duplicados y partición
El conjunto contiene imágenes repetidas o casi idénticas; si una copia queda en entrenamiento y otra en prueba, la métrica se infla. Primero se eliminaron los duplicados exactos de píxeles (#v.exact_dups imágenes). Después, las imágenes de cada clase se recorrieron en orden aleatorio (semilla #v.seed) y se aceptaron de forma voraz: una candidata se descartó si su distancia RMS de píxeles en $[0, 1]$ a cualquier imagen ya aceptada, de cualquier clase y partición, era menor o igual que #v.rms_threshold. Este umbral equivale a unos #v.rms_threshold_gray niveles de gris de diferencia media, muy por debajo de la distancia típica entre dos imágenes tomadas al azar, y cubre con margen las copias casi idénticas. Primero se llenó la prueba, clase por clase; después, con el mismo procedimiento, entrenamiento y validación: de las imágenes aceptadas de cada clase, las primeras #v.val_pc fueron a validación y las siguientes #v.train_pc a entrenamiento. Así, las #v.train_pc imágenes de entrenamiento de No-Anomaly son una muestra aleatoria de sus #v.n_noanomaly. Se rechazaron #v.rejected_total candidatas, y la distancia mínima entre imágenes de particiones distintas quedó en #v.min_cross_rms.

La prueba se formó solo con imágenes no consultadas durante el desarrollo previo del proyecto, porque una evaluación anterior había reutilizado su conjunto de prueba (exactitud #v.hist_acc con regresión logística, en otra partición y no comparable). La partición final es equilibrada (@tab:splits). El equilibrio permite comparar clases con el mismo peso, pero no refleja la frecuencia real: entre las cuatro clases elegidas, No-Anomaly representa el #v.prev_four~% del conjunto original. El conjunto no incluye identificadores de planta ni de vuelo, de modo que no puede garantizarse que entrenamiento y prueba provengan de instalaciones distintas.

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

= Metodología

== Modelos candidatos
La *regresión logística* multinomial sobre los 960 píxeles (#v.params_lr parámetros) sirve como línea base: es un modelo lineal que asigna un peso fijo a cada posición, por lo que no puede reconocer el mismo punto caliente en lugares distintos del módulo. Se configuró con $C = 1$, regularización L2 y el optimizador L-BFGS, sobre los píxeles en $[0, 1]$, sin estandarización por imagen ni aumento de datos.

Las *redes convolucionales* (CNN) aplican los mismos filtros en todas las posiciones de la imagen y combinan patrones locales en niveles sucesivos @lecun1998. Esta propiedad se ajusta al problema: un punto caliente es un patrón local cuya posición varía, y la diferencia entre clases depende de la forma y el número de regiones calientes. Se diseñó una red pequeña acorde con imágenes de 960 píxeles y con #v.train_total ejemplos de entrenamiento: tres convoluciones $3 times 3$ con ReLU (relleno 1), un _max-pooling_ $2 times 2$ después de las dos primeras, _pooling_ promedio global, _dropout_ y una capa densa con _softmax_ a cuatro clases, sin normalización por lotes. Con ancho base $w$, las convoluciones tienen $w$, $2w$ y $4w$ filtros; el mapa pasa de $40 times 24$ a $20 times 12$ y a $10 times 6$, y cada unidad de la última capa observa una ventana de $18 times 18$ píxeles. Se evaluaron dos anchos: w8 (#v.params_w8 parámetros) y w16 (#v.params_w16). Las convoluciones son equivariantes a la traslación (un patrón desplazado produce un mapa desplazado) y el _pooling_ global hace que la salida sea aproximadamente invariante a la posición del patrón.

La comparación entre ambos modelos no aísla el efecto de la arquitectura: la CNN recibe además la estandarización por imagen y el aumento por volteos, que la regresión no usa. Por ello, la mejora observada se atribuye al conjunto de arquitectura y preprocesamiento, no solo a la convolución.

== Entrenamiento
La @tab:hparams resume la configuración. Los volteos horizontales y verticales son una forma de aumento de datos coherente con el problema, porque la clase no depende de la orientación del módulo en la imagen. No se hizo búsqueda de hiperparámetros: la configuración se fijó en el protocolo y se aplicó igual a ambos anchos. La regresión logística se entrenó con scikit-learn #v.sklearn @pedregosa2011 y las CNN con PyTorch #v.torch @paszke2019 en CPU, unos #v.w16_seconds~s por entrenamiento para w16.

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
    [Optimizador], [AdamW @loshchilov2019, _weight decay_ #v.wd#[;] tasa OneCycle con máximo #v.lr],
    [Pérdida], [Entropía cruzada, suavizado de etiquetas #v.smoothing],
    [Regularización], [_Dropout_ #v.dropout antes de la capa densa],
    [Lote y épocas], [#v.batch imágenes; #v.epochs épocas, sin _early stopping_],
    [Pesos finales], [Los de la última época],
    [Semillas], [#v.seeds],
    table.hline(),
  ),
  caption: [Configuración de entrenamiento de las CNN],
) <tab:hparams>

== Protocolo de selección
Antes de entrenar se registraron los candidatos, la métrica y las reglas de decisión. Cada CNN se entrenó con #v.n_seeds semillas (#v.seeds) para estimar la variabilidad del entrenamiento; la regresión logística es determinista. Se seleccionaría la arquitectura con mayor F1 macro medio en validación; si w8 quedaba a #v.tie_margin o menos de w16, se elegiría w8 por ser más barata. El modelo desplegado sería la ejecución con semilla #v.seed de la arquitectura elegida, fijada de antemano en lugar de escoger la mejor semilla. La prueba se evaluaría una sola vez, solo para el modelo seleccionado y para la línea base como referencia.

== Evaluación
Se informan exactitud, precisión, _recall_ y F1 por clase, y sus promedios macro, que dan igual peso a cada clase. La matriz de confusión muestra qué clases se confunden. Para las curvas ROC, cada clase se trata como positiva frente a las otras tres (uno contra el resto) usando la probabilidad que el modelo asigna a esa clase; el AUC de cada curva estima la probabilidad de que una imagen de la clase reciba mayor puntuación que una de otra clase @fawcett2006. El AUC macro es el promedio simple de los cuatro valores. El AUC no depende de un umbral, mientras que la exactitud y el F1 se calculan con la clase de mayor probabilidad; por eso un AUC alto puede coexistir con confusiones apreciables.

La incertidumbre por tamaño de muestra se estimó con intervalos _bootstrap_ percentiles del 95 %: se remuestrearon con reemplazo las #v.test_total imágenes de prueba #v.resamples veces y se recalculó cada métrica @efron1993. Por último, se derivó una vista binaria para uso operativo: una predicción de cualquier clase distinta de No-Anomaly se considera alarma de anomalía, y se calculan su sensibilidad y especificidad.

= Resultados

== Selección en validación
La @tab:validation resume la validación. Ambas CNN superan con amplitud a la regresión logística. CNN w16 supera a CNN w8 por #v.w16_w8_gap en F1 macro medio, más que el margen de #v.tie_margin y que la variación entre semillas, de modo que, según el protocolo, se seleccionó *CNN w16*. La ejecución desplegada, con semilla #v.seed, obtuvo un F1 macro de validación de #v.w16_seed_f1#[;] todas las cifras de prueba corresponden a esa única ejecución.

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
En el conjunto de prueba nuevo, CNN w16 obtuvo una exactitud de #v.acc (IC 95 %: #v.acc_ci), un F1 macro de #v.f1 #v.f1_ci y un AUC macro de #v.auc #v.auc_ci. La precisión y el _recall_ macro fueron #v.prec y #v.rec. Los valores están cerca de los de validación, lo que es compatible con que la selección no haya sobreajustado esa partición. La regresión logística, evaluada en las mismas imágenes, obtuvo #v.base_acc de exactitud y #v.base_f1 #v.base_f1_ci de F1 macro (@tab:test). Los intervalos no se solapan, un criterio conservador para afirmar que la diferencia no se debe al muestreo de la prueba; no se calculó un intervalo pareado de la diferencia porque no se conservaron las predicciones por imagen de ambos modelos.

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
  caption: [Prueba (#v.test_total imágenes). Sensibilidad y especificidad de la vista binaria anomalía/normal],
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

La @tab:classes y la @fig:confusion muestran diferencias claras entre clases. *Diode* es la más fácil (F1 #v.diode_f1, _recall_ #v.diode_recall): su patrón es una franja caliente que ocupa una fracción grande del módulo y se distingue incluso a baja resolución. *No-Anomaly* tiene un _recall_ alto (#v.normal_recall) y una precisión menor (#v.normal_prec), porque también recibe anomalías que el modelo no detecta. Las clases difíciles son *Cell* (F1 #v.cell_f1) y *Cell-Multi* (F1 #v.multi_f1). Entre ellas se producen #v.cell_confusion de los #v.errors_total errores (#v.cell_confusion_share~%): #v.cell_to_multi imágenes de Cell se predijeron como Cell-Multi y #v.multi_to_cell en sentido contrario. El segundo tipo de error es una anomalía pequeña clasificada como normal (#v.cell_to_normal de Cell y #v.multi_to_normal de Cell-Multi).

Estas confusiones son coherentes con la resolución de la entrada. Ambas clases se definen por el número de celdas afectadas; en un módulo de 60 celdas, cada celda ocuparía unos #box[4 × 4] píxeles, de modo que dos celdas calientes contiguas pueden verse como una sola mancha alargada y un punto caliente débil apenas se separa del fondo. La @fig:errors ilustra los errores con mayor probabilidad asignada: las imágenes de Cell tomadas por Cell-Multi muestran una mancha grande o alargada; las de Cell-Multi tomadas por Cell muestran una sola región caliente pequeña en el borde; y las anomalías tomadas por normales tienen un contraste bajo. Algunas de estas imágenes son ambiguas también para un observador humano, por lo que parte del error puede deberse al etiquetado.

#figure(
  image("figures/error_examples.png", width: 100%),
  caption: [Errores de prueba con mayor probabilidad asignada, por tipo: clase real → predicha y probabilidad de la clase predicha (NoA: No-Anomaly; #box[C-M]: Cell-Multi).],
) <fig:errors>

Las curvas ROC (@fig:roc) confirman este orden. Diode (AUC #v.diode_auc) y No-Anomaly (#v.normal_auc) se separan bien del resto, mientras que Cell (#v.cell_auc) y Cell-Multi (#v.multi_auc) tienen curvas más bajas; la matriz de confusión muestra que buena parte de su solapamiento es mutuo. El AUC de la CNN supera al de la regresión logística en las cuatro clases. Frente a la línea base, el F1 de No-Anomaly sube de #v.base_normal_f1 a #v.normal_f1, porque la regresión confundía muchas imágenes normales con anomalías de celda, y el de Cell y Cell-Multi pasa de #v.base_cell_f1 y #v.base_multi_f1 a #v.cell_f1 y #v.multi_f1. La mejora es coherente con la equivarianza de la convolución y el _pooling_ global, aunque no se aisló su efecto del de la estandarización y el aumento de datos.

#figure(
  image("figures/roc_curves.png", width: 92%),
  caption: [Curvas ROC uno contra el resto en prueba. Líneas continuas: CNN w16 (AUC en la leyenda); punteadas: regresión logística, mismo color por clase.],
) <fig:roc>

== Utilidad y límites
Para priorizar inspecciones, la vista binaria es la más relevante. La CNN marcó como anómalas #v.detected de las #v.positives imágenes con anomalía (sensibilidad #v.sens) y generó #v.false_alarms alarmas en #v.negatives imágenes normales (especificidad #v.spec); la línea base tenía una especificidad de #v.base_spec. Con #v.missed anomalías no detectadas en prueba, el modelo sirve para ordenar la revisión, no para descartar módulos sin mirarlos. Estas proporciones se midieron con clases equilibradas. Si se trasladan a la frecuencia del conjunto original (#v.prev_four~% de imágenes normales entre las cuatro clases) suponiendo la misma sensibilidad y especificidad, alrededor del #v.ppv_field~% de las alarmas correspondería a anomalías reales y el #v.npv_field~% de las imágenes declaradas normales lo sería; con otras frecuencias estas cifras cambian. Además, en una planta real aparecerían también las otras ocho clases del conjunto, que el modelo no cubre; los propios autores advierten que las proporciones del conjunto no reproducen las de campo @millendorf2020.

Los resultados tienen límites claros. Proceden de un único conjunto de datos, de origen y cámara comunes, sin identificadores de planta, por lo que no miden la generalización a otras instalaciones. La resolución de #box[24 × 40] píxeles limita la distinción entre una y varias celdas. Solo se consideran cuatro de las doce clases: una imagen de otra categoría, como sombra o suciedad, se asignará a una de las cuatro. Las probabilidades no están calibradas y no deben leerse como confianza diagnóstica.

= Aplicación desarrollada

La aplicación permite cargar una imagen térmica o un lote de hasta 20 en formato PNG, JPEG o WebP, de #box[24 × 40] píxeles y hasta 5~MB cada una, y muestra para cada imagen la vista previa, la clase estimada y la probabilidad de cada clase, además de un resumen del lote. La @fig:app muestra parte del panel de resultados. La API no guarda las imágenes recibidas.

#figure(
  image("figures/app_result.png", width: 92%),
  caption: [Probabilidades por clase en la aplicación pública para una imagen de prueba de Cell-Multi, clasificada correctamente como «Anomalía en varias celdas». Captura recortada del panel de resultados.],
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

La @fig:arch resume la arquitectura. La interfaz, escrita en Vue 3 y TypeScript, comprueba el tipo, el tamaño y las dimensiones de cada archivo, lo decodifica en el navegador y calcula los mismos 960 valores de escala de grises usados en el entrenamiento. La API, en FastAPI, se ejecuta en un Worker de Python de Cloudflare, que también sirve la interfaz como recursos estáticos @cloudflarepython @cloudflarefastapi @cloudflareassets. El servidor valida la longitud y el rango del vector, aplica la estandarización por imagen y ejecuta la CNN.

Como PyTorch no está disponible en el Worker, los pesos se exportan a JSON (#v.artifact_kb~kB) y el paso hacia adelante se implementó en numpy. Sobre las #v.test_total imágenes de prueba, la diferencia máxima de probabilidad frente a PyTorch fue de #box($#v.parity_m times 10^(#v.parity_e)$) y la clase predicha coincidió en el #v.agreement~% de los casos. La aplicación pública, en #box(link("https://solar-vision.michaellan-eng.workers.dev/")[solar-vision.michaellan-eng.workers.dev]), sirve el modelo evaluado en este informe: la huella SHA-256 de los pesos que informa la API coincide con la del modelo exportado, y la predicción de una imagen de referencia coincide con la obtenida en el entrenamiento.

= Conclusiones

Se desarrolló un sistema completo, desde la preparación de datos hasta una aplicación web, para clasificar cuatro patrones térmicos de módulos fotovoltaicos. Una red convolucional de #v.params_w16 parámetros alcanzó en una prueba nueva un F1 macro de #v.f1 y un AUC macro de #v.auc, con una mejora clara sobre la regresión logística (#v.base_f1 de F1 macro) e intervalos de confianza que no se solapan. Como la red usó además estandarización por imagen y aumento de datos, la mejora corresponde al conjunto de arquitectura y preprocesamiento. La deduplicación entre particiones y la evaluación única de la prueba, definidas antes de entrenar, reducen el riesgo de sobreestimar el rendimiento dentro de este conjunto de datos.

El modelo reconoce de forma fiable el patrón de diodo y la ausencia de anomalía, y como detector binario alcanza una sensibilidad de #v.sens. En cambio, distinguir una celda caliente de varias sigue siendo difícil a #box[24 × 40] píxeles. Como trabajo futuro se propone evaluar el modelo con imágenes de otras plantas identificadas, usar imágenes de mayor resolución o con temperatura radiométrica, calibrar las probabilidades e incluir las demás clases del conjunto, en especial sombras y suciedad, que hoy se asignarían a alguna de las cuatro categorías.
