from collections import defaultdict
import csv
import json
import pandas as pd
import datetime

from pathlib import Path

from src.definitions import REPORTS_FOLDER


def create_model_name(prefix: str, args: dict[int, int]) -> str:
    return f"{prefix}_{"_".join([f"{k}-{v}" for k, v in args.items()])}"


def report_run(
    model_name: str,
    model_category: str,
    run_params: object,
    run_metrics: dict[str, float],
    run_desc: str = "",
    trainer_state: object = None,
    report_path: Path = REPORTS_FOLDER / "runs",
) -> Path:
    report_path.mkdir(parents=True, exist_ok=True)
    full_report_path = report_path / f"{model_name}.csv"

    params_json = json.dumps(run_params)
    trainer_json = json.dumps(trainer_state) if trainer_state else None

    row = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_category": model_category,
        "run_desc": run_desc,
        "params": params_json,
        "trainer_state": trainer_json,
        **run_metrics,
    }

    write_header = not full_report_path.exists()

    with open(full_report_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return full_report_path


def read_report(model_name: str, report_path=REPORTS_FOLDER / "runs"):
    full_report_path = report_path / f"{model_name}.csv"

    return pd.read_csv(full_report_path)


def read_report_to_show(
    model_name: str,
    sort_by_metric: str = None,
    report_path=REPORTS_FOLDER / "runs",
    model_category: str = None,
):
    report = read_report(model_name, report_path)

    report.drop("trainer_state", axis=1, inplace=True)

    if sort_by_metric:
        report = report.sort_values(by=sort_by_metric, ascending=False)

    if model_category:
        report = report[report["model_category"] == model_category]

        report.drop("model_category", axis=1, inplace=True)

    return report


def lightning_metrics(metrics):
    metric = metrics[-1]
    return {
        "loss": metric["val_loss"],
        "f1": metric["val_f1"],
        "OA": metric["val_overall_accuracy"],
        "AA": metric["val_average_accuracy"],
        "kappa": metric["val_kappa"],
    }


def classification_trainer(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss": metrics.get("eval_loss"),
        "f1": metrics.get("eval_f1"),
        "OA": metrics.get("eval_accuracy_overall"),
        "AA": metrics.get("eval_accuracy_avg"),
        "kappa": metrics.get("eval_kappa"),
    }
