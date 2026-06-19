import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_KEYS = {
    "train_loss": "Train loss",
    "train_accuracy": "Train accuracy",
    "val_loss": "Val loss",
    "val_accuracy": "Val accuracy",
    "lr": "LR",
}


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    run_files = sorted(run_dir.glob("*.json"))
    if not run_files:
        raise FileNotFoundError(f"No JSON run files found in {run_dir}")

    runs = []
    for run_file in run_files:
        with run_file.open("r", encoding="utf-8") as f:
            run = json.load(f)

        required_keys = {"Alpha", "Model", "Optimizer", "Seed"}
        missing_keys = required_keys - set(run)
        if missing_keys:
            raise ValueError(f"{run_file} is missing keys: {sorted(missing_keys)}")

        run["_source_file"] = str(run_file)
        runs.append(run)

    return runs


def group_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str, float], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for run in runs:
        key = (run["Model"], run["Optimizer"], float(run["Alpha"]))
        grouped[key].append(run)
    return dict(grouped)


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def summarize_values(values: list[float]) -> dict[str, float | None]:
    values = finite_values(values)
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}

    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def curve_stats(runs: list[dict[str, Any]], metric_key: str) -> dict[str, list[float]]:
    curves = [run[metric_key] for run in runs if metric_key in run]
    if not curves:
        return {"mean": [], "std": []}

    min_len = min(len(curve) for curve in curves)
    if min_len == 0:
        return {"mean": [], "std": []}

    mean_curve = []
    std_curve = []
    for epoch_idx in range(min_len):
        epoch_values = [float(curve[epoch_idx]) for curve in curves]
        stats = summarize_values(epoch_values)
        mean_curve.append(stats["mean"])
        std_curve.append(stats["std"])

    return {"mean": mean_curve, "std": std_curve}


def final_metric(run: dict[str, Any], metric_key: str) -> float:
    if "final" in run and metric_key in run["final"]:
        return float(run["final"][metric_key])

    if metric_key not in run or not run[metric_key]:
        raise ValueError(f"{run['_source_file']} has no metric data for {metric_key}")

    return float(run[metric_key][-1])


def best_metric(run: dict[str, Any], metric_key: str, higher_is_better: bool) -> float:
    if metric_key not in run or not run[metric_key]:
        raise ValueError(f"{run['_source_file']} has no metric data for {metric_key}")

    values = [float(value) for value in run[metric_key]]
    return max(values) if higher_is_better else min(values)


def summarize_group(
    model: str,
    optimizer: str,
    alpha: float,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    runs = sorted(runs, key=lambda run: run["Seed"])

    final_values = {
        metric_name: [final_metric(run, metric_key) for run in runs]
        for metric_name, metric_key in METRIC_KEYS.items()
        if metric_key != "LR" and any(metric_key in run for run in runs)
    }
    best_values = {
        "best_val_accuracy": [
            best_metric(run, "Val accuracy", higher_is_better=True) for run in runs
        ],
        "best_val_loss": [
            best_metric(run, "Val loss", higher_is_better=False) for run in runs
        ],
    }

    histories = {
        metric_name: curve_stats(runs, metric_key)
        for metric_name, metric_key in METRIC_KEYS.items()
        if any(metric_key in run for run in runs)
    }

    return {
        "Model": model,
        "Optimizer": optimizer,
        "Alpha": alpha,
        "Seeds": [run["Seed"] for run in runs],
        "Num runs": len(runs),
        "Source files": [run["_source_file"] for run in runs],
        "Final": {
            metric_name: summarize_values(values)
            for metric_name, values in final_values.items()
        },
        "Best": {
            metric_name: summarize_values(values)
            for metric_name, values in best_values.items()
        },
        "History": histories,
        "Runs": [
            {
                "Seed": run["Seed"],
                "final_val_accuracy": final_metric(run, "Val accuracy"),
                "final_val_loss": final_metric(run, "Val loss"),
                "best_val_accuracy": best_metric(
                    run, "Val accuracy", higher_is_better=True
                ),
                "best_val_loss": best_metric(run, "Val loss", higher_is_better=False),
            }
            for run in runs
        ],
    }


def write_json(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(output_path: Path, summaries: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model",
        "Optimizer",
        "Alpha",
        "Seeds",
        "Num runs",
        "final_val_accuracy_mean",
        "final_val_accuracy_std",
        "final_val_loss_mean",
        "final_val_loss_std",
        "best_val_accuracy_mean",
        "best_val_accuracy_std",
        "best_val_loss_mean",
        "best_val_loss_std",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            final_val_accuracy = summary["Final"]["val_accuracy"]
            final_val_loss = summary["Final"]["val_loss"]
            best_val_accuracy = summary["Best"]["best_val_accuracy"]
            best_val_loss = summary["Best"]["best_val_loss"]

            writer.writerow(
                {
                    "Model": summary["Model"],
                    "Optimizer": summary["Optimizer"],
                    "Alpha": summary["Alpha"],
                    "Seeds": " ".join(str(seed) for seed in summary["Seeds"]),
                    "Num runs": summary["Num runs"],
                    "final_val_accuracy_mean": final_val_accuracy["mean"],
                    "final_val_accuracy_std": final_val_accuracy["std"],
                    "final_val_loss_mean": final_val_loss["mean"],
                    "final_val_loss_std": final_val_loss["std"],
                    "best_val_accuracy_mean": best_val_accuracy["mean"],
                    "best_val_accuracy_std": best_val_accuracy["std"],
                    "best_val_loss_mean": best_val_loss["mean"],
                    "best_val_loss_std": best_val_loss["std"],
                }
            )


def print_table(summaries: list[dict[str, Any]]) -> None:
    header = (
        "Model",
        "Optimizer",
        "Alpha",
        "Seeds",
        "Final Acc",
        "Best Acc",
        "Final Loss",
    )
    print(
        f"{header[0]:<10} {header[1]:<10} {header[2]:>8} "
        f"{header[3]:<10} {header[4]:>18} {header[5]:>18} {header[6]:>18}"
    )
    print("-" * 98)

    for summary in summaries:
        final_acc = summary["Final"]["val_accuracy"]
        best_acc = summary["Best"]["best_val_accuracy"]
        final_loss = summary["Final"]["val_loss"]
        seeds = ",".join(str(seed) for seed in summary["Seeds"])

        print(
            f"{summary['Model']:<10} {summary['Optimizer']:<10} "
            f"{summary['Alpha']:>8g} {seeds:<10} "
            f"{final_acc['mean']:>8.4f} +/- {final_acc['std']:<7.4f} "
            f"{best_acc['mean']:>8.4f} +/- {best_acc['std']:<7.4f} "
            f"{final_loss['mean']:>8.4f} +/- {final_loss['std']:<7.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average experiment outputs across seeds."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/runs"),
        help="Directory containing per-seed JSON run files.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs/evaluation_summary.json"),
        help="Path for detailed JSON summary.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("outputs/evaluation_summary.csv"),
        help="Path for compact CSV summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = load_runs(args.run_dir)
    grouped_runs = group_runs(runs)

    summaries = [
        summarize_group(model, optimizer, alpha, group)
        for (model, optimizer, alpha), group in grouped_runs.items()
    ]
    summaries.sort(key=lambda item: (item["Model"], item["Optimizer"], item["Alpha"]))

    output = {
        "Run dir": str(args.run_dir),
        "Num run files": len(runs),
        "Num groups": len(summaries),
        "Summary": summaries,
    }

    write_json(args.json_out, output)
    write_csv(args.csv_out, summaries)
    print_table(summaries)
    print(f"\nSaved JSON: {args.json_out}")
    print(f"Saved CSV:  {args.csv_out}")


if __name__ == "__main__":
    main()
