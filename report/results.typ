#let evaluation_summary = [En la prueba reservada, el modelo de regresión logística obtuvo accuracy 0.558 y F1 macro 0.561. El despliegue público aún requiere verificación.]

#let results_body = [
El modelo seleccionado por F1 macro de validación fue *regresión logística*. En la prueba reservada (n = 240), obtuvo accuracy 0.558, precisión macro 0.564, recall macro 0.558 y F1 macro 0.561. El manifiesto contiene 1120 imágenes de entrenamiento, 240 de validación y 240 de prueba. Se retiraron 2 duplicados exactos antes de muestrear. La auditoría no encontró parejas con distancia dHash ≤ 2 entre particiones. Esto controla esa medida de similitud, pero no prueba independencia por instalación.

El AUC macro uno contra el resto fue 0.797.

#figure(table(columns: 4, inset: 3pt, [Clase], [Precisión], [Recall], [F1], [No-Anomaly], [0.550], [0.550], [0.550], [Cell], [0.469], [0.500], [0.484], [Cell-Multi], [0.450], [0.450], [0.450], [Diode], [0.786], [0.733], [0.759], ), caption: [Métricas por clase en el conjunto de prueba.]) <tab:classes>

La @tab:classes presenta los resultados por categoría. AUC uno contra el resto por clase: No-Anomaly 0.784; Cell 0.775; Cell-Multi 0.719; Diode 0.912; 

regresión logística registró F1 macro de validación 0.539 y tiempo de entrenamiento 0.545 s. MLP de 32 unidades registró F1 macro de validación 0.427 y tiempo de entrenamiento 0.344 s. La selección utilizó validación, sin ajustar hiperparámetros con el conjunto de prueba.

La confusión individual más frecuente fue Cell-Multi clasificada como Cell (21 imágenes). La @fig:confusion permite examinar las demás confusiones. La @fig:roc muestra la capacidad de separación por categoría.

El recall más bajo correspondió a Cell-Multi (0.450), mientras que el más alto fue Diode (0.733). La muestra de prueba está equilibrada y estas diferencias no deben extrapolarse a frecuencias reales de anomalías en una instalación.

#figure(image("figures/class_examples.png", width: 100%), caption: [Ejemplos de las cuatro categorías seleccionadas.]) <fig:examples>
#figure(image("figures/class_distribution.png", width: 100%), caption: [Distribución de las imágenes seleccionadas.]) <fig:distribution>
#figure(image("figures/confusion_matrix.png", width: 100%), caption: [Matriz de confusión del modelo elegido en prueba.]) <fig:confusion>
#figure(image("figures/roc_curves.png", width: 100%), caption: [Curvas ROC uno contra el resto en prueba.]) <fig:roc>
#figure(image("figures/model_comparison.png", width: 100%), caption: [Comparación de candidatos sobre validación.]) <fig:comparison>
]

#let conclusion_status = [La evaluación reservada mostró F1 macro 0.561 para regresión logística; su interpretación debe considerar las cuatro clases elegidas y la posible similitud entre capturas de una misma instalación. La paridad de decodificación JPEG y el runtime de Workers aún requieren verificación independiente.]
