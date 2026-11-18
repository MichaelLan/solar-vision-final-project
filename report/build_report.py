import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / "artifacts"
METRICS = ARTIFACTS / "metrics.json"
MANIFEST = ARTIFACTS / "manifest.json"
OUTPUT = ROOT / "results.typ"
CLASSES = ("No-Anomaly", "Cell", "Cell-Multi", "Diode")
MODEL_NAMES = {"logistic_regression": "regresión logística", "mlp_32": "MLP de 32 unidades"}


def number(value: object) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite metric")
    return f"{result:.3f}"


def pending() -> str:
    return """#let evaluation_summary = [El ZIP original fue obtenido y se verificaron sus metadatos. El entrenamiento y la evaluación todavía no han producido métricas verificadas; el despliegue público también permanece pendiente.]

#let results_body = [
  Se verificaron 20 000 registros en el JSON del ZIP original. El entrenamiento y la evaluación están en curso; todavía no existen valores verificados de accuracy, precisión, recall, F1 o AUC. No se incluyen tablas ni curvas sin medición. Esta sección se genera de nuevo al completar el pipeline reproducible.
]

#let conclusion_status = [La validez cuantitativa depende de completar el entrenamiento con el archivo original y la prueba de paridad entre navegador, entrenamiento y Worker.]
"""


def figure(filename: str, caption: str, label: str) -> str:
    if not (ROOT / "figures" / filename).is_file():
        raise FileNotFoundError(f"Missing figure required by metrics: {filename}")
    return f'#figure(image("figures/{filename}", width: 100%), caption: [{caption}]) <{label}>\n'


def measured(data: dict) -> str:
    chosen = data["selected_model"]
    test = chosen["test"]
    models = data["models"]
    name = chosen["name"]
    display_name = MODEL_NAMES.get(name, name)
    if name not in models or any(label not in test["per_class"] for label in CLASSES):
        raise ValueError("Incomplete model metrics")
    body = [
        f"El modelo seleccionado por F1 macro de validación fue *{display_name}*. ",
        f"En la prueba reservada (n = {int(test['count'])}), obtuvo accuracy {number(test['accuracy'])}, ",
        f"precisión macro {number(test['precision_macro'])}, recall macro {number(test['recall_macro'])} ",
        f"y F1 macro {number(test['f1_macro'])}. ",
    ]
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        splits = manifest["split_counts_by_class"]
        if all(label in splits for label in CLASSES):
            train_count = sum(int(splits[label]["train"]) for label in CLASSES)
            validation_count = sum(int(splits[label]["validation"]) for label in CLASSES)
            test_count = sum(int(splits[label]["test"]) for label in CLASSES)
            body.append(
                f"El manifiesto contiene {train_count} imágenes de entrenamiento, "
                f"{validation_count} de validación y {test_count} de prueba. "
                f"Se retiraron {int(manifest['exact_duplicates_removed_before_sampling'])} "
                "duplicados exactos antes de muestrear. "
            )
            near_cross = int(manifest["dhash_cross_split_pairs_hamming_le_2"])
            if near_cross == 0:
                body.append(
                    "La auditoría no encontró parejas con distancia dHash ≤ 2 entre "
                    "particiones. Esto controla esa medida de similitud, pero no prueba "
                    "independencia por instalación.\n\n"
                )
            else:
                body.append(
                    f"La auditoría detectó {near_cross} parejas con distancia dHash ≤ 2 "
                    "en particiones distintas. Es una limitación de independencia visual, "
                    "no una estimación de imágenes duplicadas ni de sitios compartidos.\n\n"
                )
    if test["auc_macro_ovr"] is not None:
        body.append(f"El AUC macro uno contra el resto fue {number(test['auc_macro_ovr'])}.\n\n")
    else:
        body.append("El AUC macro no pudo calcularse para todas las clases.\n\n")
    body.extend([
        "#figure(table(columns: 4, inset: 3pt, ",
        "[Clase], [Precisión], [Recall], [F1], ",
    ])
    for label in CLASSES:
        values = test["per_class"][label]
        body.append(
            f"[{label}], [{number(values['precision'])}], [{number(values['recall'])}], [{number(values['f1'])}], "
        )
    body.append("), caption: [Métricas por clase en el conjunto de prueba.]) <tab:classes>\n\n")
    body.append("La @tab:classes presenta los resultados por categoría. ")
    body.append("AUC uno contra el resto por clase: ")
    for label in CLASSES:
        auc_value = test["per_class"][label]["auc_ovr"]
        body.append(f"{label} {number(auc_value) if auc_value is not None else 'no calculable'}; ")
    body.append("\n\n")
    for candidate_name, candidate in models.items():
        body.append(
            f"{MODEL_NAMES.get(candidate_name, candidate_name)} registró F1 macro de validación {number(candidate['validation']['f1_macro'])} "
            f"y tiempo de entrenamiento {number(candidate['training_seconds'])} s. "
        )
    body.append("La selección utilizó validación, sin ajustar hiperparámetros con el conjunto de prueba.\n\n")
    matrix = test["confusion_matrix"]
    confusions = sorted(
        ((int(matrix[left][right]), CLASSES[left], CLASSES[right])
         for left in range(len(CLASSES)) for right in range(len(CLASSES)) if left != right),
        reverse=True,
    )
    if confusions:
        count, actual, predicted = confusions[0]
        body.append(
            f"La confusión individual más frecuente fue {actual} clasificada como {predicted} "
            f"({count} imágenes). La @fig:confusion permite examinar las demás confusiones. "
            "La @fig:roc muestra la capacidad de separación por categoría.\n\n"
        )
    by_recall = sorted(
        ((float(test["per_class"][label]["recall"]), label) for label in CLASSES)
    )
    lowest_recall, lowest_label = by_recall[0]
    highest_recall, highest_label = by_recall[-1]
    body.append(
        f"El recall más bajo correspondió a {lowest_label} ({number(lowest_recall)}), "
        f"mientras que el más alto fue {highest_label} ({number(highest_recall)}). "
        "La muestra de prueba está equilibrada y estas diferencias no deben extrapolarse "
        "a frecuencias reales de anomalías en una instalación.\n\n"
    )
    figures = [
        ("class_examples.png", "Ejemplos de las cuatro categorías seleccionadas.", "fig:examples"),
        ("class_distribution.png", "Distribución de las imágenes seleccionadas.", "fig:distribution"),
        ("confusion_matrix.png", "Matriz de confusión del modelo elegido en prueba.", "fig:confusion"),
        ("roc_curves.png", "Curvas ROC uno contra el resto en prueba.", "fig:roc"),
        ("model_comparison.png", "Comparación de candidatos sobre validación.", "fig:comparison"),
    ]
    for filename, caption, label in figures:
        body.append(figure(filename, caption, label))
    summary = (
        f"En la prueba reservada, el modelo de {display_name} obtuvo accuracy {number(test['accuracy'])} "
        f"y F1 macro {number(test['f1_macro'])}. El despliegue público aún requiere verificación."
    )
    conclusion = (
        f"La evaluación reservada mostró F1 macro {number(test['f1_macro'])} para {display_name}; "
        "su interpretación debe considerar las cuatro clases elegidas y la posible similitud entre capturas de una misma instalación. "
        "La paridad de decodificación JPEG y el runtime de Workers aún requieren verificación independiente."
    )
    return (
        f"#let evaluation_summary = [{summary}]\n\n"
        + "#let results_body = [\n"
        + "".join(body)
        + "]\n\n"
        + f"#let conclusion_status = [{conclusion}]\n"
    )


def main() -> None:
    content = pending() if not METRICS.is_file() else measured(json.loads(METRICS.read_text(encoding="utf-8")))
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Generated {OUTPUT} ({'measured' if METRICS.is_file() else 'pending'})")


if __name__ == "__main__":
    main()
