from pathlib import Path
import sys
import re
import csv

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
        "exp_type": "Custom CNN",
        "result_dir": RESULTS_DIR / "baseline_cnn",
        "test_file": "baseline_cnn_test_result.txt",
        "report_file": "baseline_cnn_classification_report.txt",
    },
    {
        "model_key": "densenet121",
        "model_name": "DenseNet121",
        "exp_type": "Transfer Learning",
        "result_dir": RESULTS_DIR / "densenet121",
        "test_file": "densenet121_test_result.txt",
        "report_file": "densenet121_classification_report.txt",
    },
    {
        "model_key": "efficientnetb0",
        "model_name": "EfficientNetB0",
        "exp_type": "Transfer Learning",
        "result_dir": RESULTS_DIR / "efficientnetb0",
        "test_file": "efficientnetb0_test_result.txt",
        "report_file": "efficientnetb0_classification_report.txt",
    },
    {
        "model_key": "mobilenetv2",
        "model_name": "MobileNetV2",
        "exp_type": "Lightweight Transfer Learning",
        "result_dir": RESULTS_DIR / "mobilenetv2",
        "test_file": "mobilenetv2_test_result.txt",
        "report_file": "mobilenetv2_classification_report.txt",
    },
    {
        "model_key": "efficientnetb0_classweight",
        "model_name": "EfficientNetB0 + Class Weight",
        "exp_type": "Imbalance Handling",
        "result_dir": RESULTS_DIR / "imbalance" / "efficientnetb0_classweight",
        "test_file": "efficientnetb0_classweight_test_result.txt",
        "report_file": "efficientnetb0_classweight_classification_report.txt",
    },
]


def read_text(path):
    if not path.exists():
        print(f"[WARNING] File not found: { path }")
        return ""
    return path.read_text(encoding="utf-8")


def parse_test_result(text):
    loss = None
    acc = None

    loss_match = re.search(r"Test Loss:\s*([0-9.]+)", text)
    acc_match = re.search(r"Test Accuracy:\s*([0-9.]+)", text)

    if loss_match:
        loss = float(loss_match.group(1))
    if acc_match:
        acc = float(acc_match.group(1))

    return loss, acc


def parse_classification_report(text):
    macro_f1 = None
    weighted_f1 = None

    macro_match = re.search(
        r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)",
        text,
    )
    weighted_match = re.search(
        r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)",
        text,
    )

    if macro_match:
        macro_f1 = float(macro_match.group(3))

    if weighted_match:
        weighted_f1 = float(weighted_match.group(3))

    return macro_f1, weighted_f1


def count_epochs(history_path):
    if not history_path.exists():
        return None

    with history_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return len(rows)


def get_decision(model_name):
    if model_name == "EfficientNetB0":
        return "Best model candidate"
    if model_name == "MobileNetV2":
        return "Lightweight model candidate"
    if model_name == "Baseline CNN":
        return "Baseline model"
    if model_name == "EfficientNetB0 + Class Weight":
        return "Useful for imbalance analysis"
    if model_name == "DenseNet121":
        return "Needs improvement"
    return ""


rows = []

for model in MODELS:
    result_dir = model["result_dir"]

    test_text = read_text(result_dir / model["test_file"])
    report_text = read_text(result_dir / model["report_file"])

    test_loss, test_acc = parse_test_result(test_text)
    macro_f1, weighted_f1 = parse_classification_report(report_text)

    history_files = list(result_dir.glob("*history.csv"))
    epoch = count_epochs(history_files[0]) if history_files else None

    rows.append({
        "Model": model["model_name"],
        "Experiment Type": model["exp_type"],
        "Epoch": epoch,
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Decision": get_decision(model["model_name"]),
    })


rows = sorted(
    rows,
    key=lambda x: x["Test Accuracy"] if x["Test Accuracy"] is not None else -1,
    reverse=True,
)

for idx, row in enumerate(rows, start=1):
    row["Rank"] = idx


csv_path = SUMMARY_DIR / "model_comparison_summary.csv"
md_path = SUMMARY_DIR / "model_comparison_summary.md"

fieldnames = [
    "Rank",
    "Model",
    "Experiment Type",
    "Epoch",
    "Test Loss",
    "Test Accuracy",
    "Macro F1",
    "Weighted F1",
    "Decision",
]

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


with md_path.open("w", encoding="utf-8") as f:
    f.write("# Model Comparison Summary\n\n")
    f.write("| Rank | Model | Experiment Type | Epoch | Test Loss | Test Accuracy | Macro F1 | Weighted F1 | Decision |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|---:|---|\n")

    for row in rows:
        f.write(
            f"| {fmt(row['Rank'])} "
            f"| {fmt(row['Model'])} "
            f"| {fmt(row['Experiment Type'])} "
            f"| {fmt(row['Epoch'])} "
            f"| {fmt(row['Test Loss'])} "
            f"| {fmt(row['Test Accuracy'])} "
            f"| {fmt(row['Macro F1'])} "
            f"| {fmt(row['Weighted F1'])} "
            f"| {fmt(row['Decision'])} |\n"
        )

print("Model comparison summary created!")
print(f"CSV saved to: {csv_path}")
print(f"Markdown saved to: {md_path}")