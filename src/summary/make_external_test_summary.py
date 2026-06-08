from pathlib import Path
import sys
import re
import csv

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_DIR = RESULTS_DIR / "external"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME_MAP = {
    "baseline": "Baseline CNN",
    "baseline_cnn": "Baseline CNN",
    "densenet121": "DenseNet121",
    "efficientnetb0": "EfficientNetB0",
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0_classweight": "EfficientNetB0 + Class Weight",
}

DATASETS = [
    {
        "dataset_key": "taiwan",
        "dataset_name": "Taiwan External",
        "result_dir": EXTERNAL_DIR / "taiwan",
    },
    {
        "dataset_key": "bangladesh_bbox",
        "dataset_name": "Bangladesh BBox External",
        "result_dir": EXTERNAL_DIR / "bangladesh_bbox",
    },
]


def parse_classification_report(text):
    class_metrics = {}
    macro_f1 = None
    weighted_f1 = None

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("macro avg"):
            match = re.match(
                r"^macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)$",
                line,
            )
            if match:
                macro_f1 = float(match.group(3))
            continue

        if line.startswith("weighted avg"):
            match = re.match(
                r"^weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)$",
                line,
            )
            if match:
                weighted_f1 = float(match.group(3))
            continue

        if line.startswith("accuracy"):
            continue

        match = re.match(
            r"^(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)$",
            line,
        )

        if match:
            class_name = match.group(1).strip()
            precision = float(match.group(2))
            recall = float(match.group(3))
            f1_score = float(match.group(4))
            support = int(match.group(5))

            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": support,
            }

    return class_metrics, macro_f1, weighted_f1


def parse_prediction_distribution(text):
    prediction_distribution = {}
    in_prediction_section = False

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Prediction Distribution"):
            in_prediction_section = True
            continue

        if not in_prediction_section or not line:
            continue

        match = re.match(r"^(.+):\s+([0-9]+)$", line)

        if match:
            class_name = match.group(1).strip()
            count = int(match.group(2))
            prediction_distribution[class_name] = count

    return prediction_distribution


def parse_external_report(path):
    text = path.read_text(encoding="utf-8")

    model_match = re.search(r"Model:\s*(.+)", text)
    acc_match = re.search(r"External Test Accuracy:\s*([0-9.]+)", text)

    model_key = model_match.group(1).strip() if model_match else path.stem
    model_name = MODEL_NAME_MAP.get(model_key, model_key)

    accuracy = float(acc_match.group(1)) if acc_match else None

    class_metrics, macro_f1, weighted_f1 = parse_classification_report(text)
    prediction_distribution = parse_prediction_distribution(text)

    support_classes = [
        class_name
        for class_name, metrics in class_metrics.items()
        if metrics["support"] > 0
    ]

    zero_recall_classes = [
        class_name
        for class_name, metrics in class_metrics.items()
        if metrics["support"] > 0 and metrics["recall"] == 0
    ]

    if support_classes:
        overlap_macro_f1 = sum(
            class_metrics[class_name]["f1-score"] for class_name in support_classes
        ) / len(support_classes)

        total_support = sum(
            class_metrics[class_name]["support"] for class_name in support_classes
        )

        overlap_weighted_f1 = sum(
            class_metrics[class_name]["f1-score"] * class_metrics[class_name]["support"]
            for class_name in support_classes
        ) / total_support
    else:
        overlap_macro_f1 = None
        overlap_weighted_f1 = None
        total_support = 0

    if prediction_distribution:
        dominant_predicted_class = max(
            prediction_distribution,
            key=prediction_distribution.get,
        )
        dominant_predicted_count = prediction_distribution[dominant_predicted_class]
    else:
        dominant_predicted_class = None
        dominant_predicted_count = None

    return {
        "model_key": model_key,
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1_10class": macro_f1,
        "weighted_f1": weighted_f1,
        "support_class_count": len(support_classes),
        "total_support": total_support,
        "overlap_macro_f1": overlap_macro_f1,
        "overlap_weighted_f1": overlap_weighted_f1,
        "zero_recall_class_count": len(zero_recall_classes),
        "zero_recall_classes": "; ".join(zero_recall_classes),
        "dominant_predicted_class": dominant_predicted_class,
        "dominant_predicted_count": dominant_predicted_count,
        "class_metrics": class_metrics,
        "prediction_distribution": prediction_distribution,
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


summary_rows = []
classwise_rows = []
prediction_rows = []

for dataset in DATASETS:
    dataset_name = dataset["dataset_name"]
    result_dir = dataset["result_dir"]

    if not result_dir.exists():
        print(f"[WARNING] External result folder not found: {result_dir}")
        continue

    report_files = sorted(result_dir.glob("*external_report.txt"))

    if not report_files:
        print(f"[WARNING] No external report found in: {result_dir}")
        continue

    dataset_rows = []

    for report_file in report_files:
        print(f"[INFO] Reading: {report_file}")

        parsed = parse_external_report(report_file)

        row = {
            "Dataset": dataset_name,
            "Model": parsed["model_name"],
            "Accuracy": parsed["accuracy"],
            "Macro F1 (10-class)": parsed["macro_f1_10class"],
            "Weighted F1": parsed["weighted_f1"],
            "Support Class Count": parsed["support_class_count"],
            "Total Support": parsed["total_support"],
            "Overlap Macro F1": parsed["overlap_macro_f1"],
            "Overlap Weighted F1": parsed["overlap_weighted_f1"],
            "Zero Recall Class Count": parsed["zero_recall_class_count"],
            "Zero Recall Classes": parsed["zero_recall_classes"],
            "Dominant Predicted Class": parsed["dominant_predicted_class"],
            "Dominant Predicted Count": parsed["dominant_predicted_count"],
        }

        dataset_rows.append(row)

        for class_name, metrics in parsed["class_metrics"].items():
            classwise_rows.append({
                "Dataset": dataset_name,
                "Model": parsed["model_name"],
                "Class": class_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1-score"],
                "Support": metrics["support"],
            })

        for class_name, count in parsed["prediction_distribution"].items():
            prediction_rows.append({
                "Dataset": dataset_name,
                "Model": parsed["model_name"],
                "Predicted Class": class_name,
                "Predicted Count": count,
            })

    dataset_rows = sorted(
        dataset_rows,
        key=lambda x: x["Accuracy"] if x["Accuracy"] is not None else -1,
        reverse=True,
    )

    for rank, row in enumerate(dataset_rows, start=1):
        row["Rank"] = rank
        summary_rows.append(row)


# 1) External summary CSV
summary_csv_path = SUMMARY_DIR / "external_test_summary.csv"

summary_fieldnames = [
    "Dataset",
    "Rank",
    "Model",
    "Accuracy",
    "Macro F1 (10-class)",
    "Weighted F1",
    "Support Class Count",
    "Total Support",
    "Overlap Macro F1",
    "Overlap Weighted F1",
    "Zero Recall Class Count",
    "Zero Recall Classes",
    "Dominant Predicted Class",
    "Dominant Predicted Count",
]

with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)


# 2) Class-wise external CSV
classwise_csv_path = SUMMARY_DIR / "external_classwise_metrics_long.csv"

with classwise_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "Dataset",
        "Model",
        "Class",
        "Precision",
        "Recall",
        "F1-score",
        "Support",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(classwise_rows)


# 3) Prediction distribution CSV
prediction_csv_path = SUMMARY_DIR / "external_prediction_distribution.csv"

with prediction_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "Dataset",
        "Model",
        "Predicted Class",
        "Predicted Count",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(prediction_rows)


# 4) Markdown summary
md_path = SUMMARY_DIR / "external_test_summary.md"

with md_path.open("w", encoding="utf-8") as f:
    f.write("# External Test Summary\n\n")
    f.write(
        "| Dataset | Rank | Model | Accuracy | Weighted F1 | "
        "Overlap Macro F1 | Overlap Weighted F1 | Dominant Prediction | Zero Recall |\n"
    )
    f.write("|---|---:|---|---:|---:|---:|---:|---|---:|\n")

    for row in summary_rows:
        dominant = row["Dominant Predicted Class"]
        dominant_count = row["Dominant Predicted Count"]

        if dominant is not None:
            dominant_text = f"{dominant} ({dominant_count})"
        else:
            dominant_text = "-"

        zero_recall_text = (
            f"{row['Zero Recall Class Count']}/{row['Support Class Count']}"
        )

        f.write(
            f"| {row['Dataset']} "
            f"| {row['Rank']} "
            f"| {row['Model']} "
            f"| {fmt(row['Accuracy'])} "
            f"| {fmt(row['Weighted F1'])} "
            f"| {fmt(row['Overlap Macro F1'])} "
            f"| {fmt(row['Overlap Weighted F1'])} "
            f"| {dominant_text} "
            f"| {zero_recall_text} |\n"
        )


print("\nExternal test summary created!")
print(f"Summary CSV saved to: {summary_csv_path}")
print(f"Class-wise CSV saved to: {classwise_csv_path}")
print(f"Prediction distribution CSV saved to: {prediction_csv_path}")
print(f"Markdown saved to: {md_path}")