from pathlib import Path
import re
import csv
import json
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "model_key": "baseline_cnn",
        "model_name": "Baseline CNN",
        "result_dir": RESULTS_DIR / "baseline_cnn",
        "report_file": "baseline_cnn_classification_report.txt",
    },
    {
        "model_key": "densenet121",
        "model_name": "DenseNet121",
        "result_dir": RESULTS_DIR / "densenet121",
        "report_file": "densenet121_classification_report.txt",
    },
    {
        "model_key": "efficientnetb0",
        "model_name": "EfficientNetB0",
        "result_dir": RESULTS_DIR / "efficientnetb0",
        "report_file": "efficientnetb0_classification_report.txt",
    },
    {
        "model_key": "mobilenetv2",
        "model_name": "MobileNetV2",
        "result_dir": RESULTS_DIR / "mobilenetv2",
        "report_file": "mobilenetv2_classification_report.txt",
    },
    {
        "model_key": "efficientnetb0_classweight",
        "model_name": "EfficientNetB0 + Class Weight",
        "result_dir": RESULTS_DIR / "imbalance" / "efficientnetb0_classweight",
        "report_file": "efficientnetb0_classweight_classification_report.txt",
    },
]


def find_report_file(result_dir, preferred_name):
    preferred_path = result_dir / preferred_name

    if preferred_path.exists():
        return preferred_path

    candidates = list(result_dir.glob("*classification_report*.txt"))
    if candidates:
        return candidates[0]

    candidates = list(result_dir.glob("*test_report*.json"))
    if candidates:
        return candidates[0]

    candidates = list(result_dir.glob("*.json"))
    if candidates:
        return candidates[0]

    return None


def parse_text_report(text):
    """
    Parse sklearn classification_report text format.
    Example line:
    Tomato___Early_blight       0.8615    0.3733    0.5209       150
    """
    class_metrics = {}

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("accuracy"):
            continue

        if line.startswith("macro avg") or line.startswith("weighted avg"):
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

    return class_metrics


def parse_json_report(path):
    """
    Parse json report format.
    Supports:
    1) direct sklearn classification_report dict
    2) {"classification_report": {...}}
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "classification_report" in data:
        data = data["classification_report"]

    class_metrics = {}

    for class_name, values in data.items():
        if class_name in ["accuracy", "macro avg", "weighted avg"]:
            continue

        if not isinstance(values, dict):
            continue

        if "f1-score" not in values:
            continue

        class_metrics[class_name] = {
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1-score": float(values.get("f1-score", 0.0)),
            "support": int(values.get("support", 0)),
        }

    return class_metrics


def load_class_metrics(report_path):
    if report_path.suffix.lower() == ".json":
        return parse_json_report(report_path)

    text = report_path.read_text(encoding="utf-8")
    return parse_text_report(text)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


all_rows = []
class_order = []

for model in MODELS:
    report_path = find_report_file(model["result_dir"], model["report_file"])

    if report_path is None:
        print(f"[WARNING] Report file not found: {model['model_name']}")
        continue

    print(f"[INFO] Reading {model['model_name']} report: {report_path}")

    class_metrics = load_class_metrics(report_path)

    for class_name, metrics in class_metrics.items():
        if class_name not in class_order:
            class_order.append(class_name)

        all_rows.append({
            "Class": class_name,
            "Model": model["model_name"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1-score"],
            "Support": metrics["support"],
        })


# 1) Long format CSV
long_csv_path = SUMMARY_DIR / "classwise_f1_long.csv"

with long_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = ["Class", "Model", "Precision", "Recall", "F1-score", "Support"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)


# 2) Pivot format: Class x Model F1-score
models = [model["model_name"] for model in MODELS]

f1_map = {}
support_map = {}

for row in all_rows:
    class_name = row["Class"]
    model_name = row["Model"]

    f1_map.setdefault(class_name, {})
    support_map.setdefault(class_name, {})

    f1_map[class_name][model_name] = row["F1-score"]
    support_map[class_name][model_name] = row["Support"]


pivot_rows = []

for class_name in class_order:
    row = {"Class": class_name}

    best_model = None
    best_f1 = -1.0

    for model_name in models:
        f1 = f1_map.get(class_name, {}).get(model_name)

        row[model_name] = f1

        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_model = model_name

    row["Best Model"] = best_model
    row["Best F1"] = best_f1 if best_f1 >= 0 else None

    pivot_rows.append(row)


pivot_csv_path = SUMMARY_DIR / "classwise_f1_comparison.csv"

pivot_fieldnames = ["Class"] + models + ["Best Model", "Best F1"]

with pivot_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=pivot_fieldnames)
    writer.writeheader()
    writer.writerows(pivot_rows)


# 3) Markdown table for report
md_path = SUMMARY_DIR / "classwise_f1_comparison.md"

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Class-wise F1-score Comparison\n\n")

    header = "| Class | " + " | ".join(models) + " | Best Model | Best F1 |\n"
    separator = "|---|" + "|".join(["---:"] * len(models)) + "|---|---:|\n"

    f.write(header)
    f.write(separator)

    for row in pivot_rows:
        class_name = row["Class"].replace("|", "/")

        line = f"| {class_name} "

        for model_name in models:
            line += f"| {fmt(row.get(model_name))} "

        line += f"| {fmt(row.get('Best Model'))} | {fmt(row.get('Best F1'))} |\n"

        f.write(line)


# 4) Weak classes of current best model
best_model_name = "EfficientNetB0"

weak_rows = []

for row in pivot_rows:
    f1 = row.get(best_model_name)

    if f1 is not None:
        weak_rows.append({
            "Class": row["Class"],
            "EfficientNetB0 F1-score": f1,
        })

weak_rows = sorted(weak_rows, key=lambda x: x["EfficientNetB0 F1-score"])

weak_csv_path = SUMMARY_DIR / "efficientnetb0_weak_classes.csv"

with weak_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = ["Class", "EfficientNetB0 F1-score"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(weak_rows)


print("\nClass-wise F1 summary created!")
print(f"Long CSV saved to: {long_csv_path}")
print(f"Comparison CSV saved to: {pivot_csv_path}")
print(f"Markdown saved to: {md_path}")
print(f"Weak class CSV saved to: {weak_csv_path}")