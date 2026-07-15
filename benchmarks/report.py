from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io import read_json


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _agreement(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_rows = {row["case_id"]: row for row in left["observations"]}
    right_rows = {row["case_id"]: row for row in right["observations"]}
    shared = sorted(set(left_rows) & set(right_rows))
    if not shared:
        return {"decision_agreement": 0.0, "top1_agreement": 0.0}
    return {
        "decision_agreement": sum(
            left_rows[key]["predicted"] == right_rows[key]["predicted"]
            for key in shared
        )
        / len(shared),
        "top1_agreement": sum(
            left_rows[key]["matched_record_id"] == right_rows[key]["matched_record_id"]
            for key in shared
        )
        / len(shared),
    }


def _render_charts(results: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for result in results:
        sweep = result.get("threshold_sweep", [])
        if sweep:
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(
                [row["recall"] for row in sweep],
                [row["precision"] for row in sweep],
                c=[row["unsafe_reuse_rate"] for row in sweep],
                cmap="viridis_r",
            )
            axis.set_xlabel("Reuse recall")
            axis.set_ylabel("Reuse precision")
            axis.set_title(f"Threshold trade-off: {result['run_name']}")
            axis.grid(alpha=0.25)
            path = chart_dir / f"{result['run_name']}-precision-recall.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            generated.append(str(path))
    if len(results) > 1:
        figure, axis = plt.subplots(figsize=(8, 5))
        labels = [result["run_name"] for result in results]
        p50 = [result["quality"]["latency_ms"]["p50"] for result in results]
        p95 = [result["quality"]["latency_ms"]["p95"] for result in results]
        positions = range(len(results))
        axis.bar([value - 0.2 for value in positions], p50, width=0.4, label="p50")
        axis.bar([value + 0.2 for value in positions], p95, width=0.4, label="p95")
        axis.set_xticks(list(positions), labels, rotation=30, ha="right")
        axis.set_ylabel("Query latency (ms)")
        axis.set_title("Remem query latency by completed run")
        axis.legend()
        path = chart_dir / "latency-comparison.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        generated.append(str(path))
    return generated


def generate_report(result_paths: list[str | Path], output: str | Path) -> Path:
    results = [read_json(path) for path in result_paths]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    charts = _render_charts(results, destination.parent)
    lines = [
        "# Remem benchmark report",
        "",
        "> Generated exclusively from the machine-readable result files listed below.",
        "",
        "## Completed runs",
        "",
        "| Run | Dataset | Records | Queries | Mode | p50 ms | p95 ms | Precision | Recall | Unsafe reuse |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        quality = result["quality"]
        workload = result["workload"]
        lines.append(
            "| {run} | {dataset} | {records} | {queries} | {mode} | {p50:.3f} | "
            "{p95:.3f} | {precision} | {recall} | {unsafe} |".format(
                run=result["run_name"],
                dataset=workload["dataset"],
                records=workload["seed_records"],
                queries=workload["queries"],
                mode=result["search"]["resolved_mode"],
                p50=quality["latency_ms"]["p50"],
                p95=quality["latency_ms"]["p95"],
                precision=_percent(quality["precision"]),
                recall=_percent(quality["recall"]),
                unsafe=_percent(quality["unsafe_reuse_rate"]),
            )
        )
    lines.extend(["", "## Exact versus HNSW agreement", ""])
    comparisons = 0
    for exact in results:
        if exact["search"]["resolved_mode"] != "exact_cosine":
            continue
        for ann in results:
            if (
                ann["search"]["resolved_mode"] == "hnsw_cosine"
                and ann["workload"]["name"] == exact["workload"]["name"]
            ):
                agreement = _agreement(exact, ann)
                lines.append(
                    f"- `{ann['run_name']}` versus `{exact['run_name']}`: "
                    f"decision agreement {_percent(agreement['decision_agreement'])}; "
                    f"final top-1 agreement {_percent(agreement['top1_agreement'])}."
                )
                comparisons += 1
    if comparisons == 0:
        lines.append("- No paired exact/HNSW runs were supplied.")

    real_results = [
        result for result in results if result["workload"]["dataset"] != "tiny_fixture"
    ]
    lines.extend(["", "## Production-readiness assessment", ""])
    if not real_results:
        lines.append(
            "**Not assessed.** Only synthetic framework smoke tests were supplied; they are not production evidence."
        )
    else:
        worst_unsafe = max(
            result["quality"]["unsafe_reuse_rate"] for result in real_results
        )
        if worst_unsafe > 0.01:
            lines.append(
                "**Not ready for production.** At least one tested workload exceeded a 1% unsafe-reuse rate."
            )
        else:
            lines.append(
                "**Suitable for controlled production pilots only.** Tested unsafe reuse was at or below 1%, "
                "but dataset, hardware, concurrency, and deployment coverage remain bounded."
            )
    lines.extend(
        [
            "",
            "## Validity threats",
            "",
            "- Dataset labels are proxies for application-specific answer safety.",
            "- Intent equality supports retrieval reuse but deliberately does not imply identical final answers.",
            "- SQuAD passage and answer-span labels model RAG reuse, not conversational follow-up behavior.",
            "- Process RSS includes Python and embedding-library allocations and is not an isolated index measurement.",
            "- Simulated cost assumptions are estimates, never observed provider billing.",
            "- Results apply only to the recorded model revision, dependencies, hardware, and configuration.",
            "",
            "## Source result files",
            "",
        ]
    )
    lines.extend(f"- `{Path(path)}`" for path in result_paths)
    if charts:
        lines.extend(["", "## Generated charts", ""])
        lines.extend(f"- `{path}`" for path in charts)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown benchmark report")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="benchmarks/results/report.md")
    args = parser.parse_args()
    path = generate_report(args.results, args.output)
    print(json.dumps({"report": str(path)}, indent=2))


if __name__ == "__main__":
    main()
