"""Build the README comparison table from per-experiment metrics JSON files."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("malaria.compare")
EXPERIMENTS = (
    ("baseline", "Baseline CNN"),
    ("regularized", "Regularized CNN"),
    ("augmented", "Augmented CNN"),
    ("transfer", "MobileNetV2 transfer"),
    ("ensemble", "Ensemble"),
)
TABLE_START = "<!-- MODEL_COMPARISON_START -->"
TABLE_END = "<!-- MODEL_COMPARISON_END -->"


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "—"


def _load_results(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        if path.name == "final_malaria_model_metrics.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = payload.get("experiment")
        if experiment in {name for name, _ in EXPERIMENTS}:
            results[experiment] = payload
    return results


def _build_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall (Parasitized) | Specificity | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, label in EXPERIMENTS:
        metrics = results.get(experiment)
        if metrics is None:
            lines.append(f"| {label} | — | — | — | — | — | — | — |")
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                _format_metric(metrics.get("accuracy")),
                _format_metric(metrics.get("precision")),
                _format_metric(metrics.get("recall")),
                _format_metric(metrics.get("specificity")),
                _format_metric(metrics.get("f1")),
                _format_metric(metrics.get("roc_auc")),
                _format_metric(metrics.get("pr_auc")),
            )
        )
    return "\n".join(lines)


def update_readme(metrics_dir: Path, readme: Path) -> None:
    """Replace only the delimited comparison table in README.md."""
    content = readme.read_text(encoding="utf-8")
    if TABLE_START not in content or TABLE_END not in content:
        raise ValueError(f"README must include {TABLE_START} and {TABLE_END} markers.")
    start = content.index(TABLE_START) + len(TABLE_START)
    end = content.index(TABLE_END)
    if end < start:
        raise ValueError("README comparison markers are out of order.")
    table = "\n" + _build_table(_load_results(metrics_dir)) + "\n"
    readme.write_text(content[:start] + table + content[end:], encoding="utf-8")
    LOGGER.info("Updated model comparison table in %s", readme)


def main() -> None:
    """Parse paths and regenerate the README model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("."))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    update_readme(args.metrics_dir, args.readme)


if __name__ == "__main__":
    main()
