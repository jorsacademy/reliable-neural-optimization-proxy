from __future__ import annotations

from pathlib import Path

from reliable_proxy.benchmark import run_benchmark, save_report_csv, save_report_json
from reliable_proxy.dataset import build_dataset
from reliable_proxy.experiment import ResearchConfig, run_research_experiment
from reliable_proxy.generator import generate_instances
from reliable_proxy.model import TrainingConfig
from reliable_proxy.training import train_proxy


def _proxy_for_benchmark():
    train = build_dataset(
        generate_instances(90, unit_count=4, regime="in_distribution", seed=100)
    )
    validation = build_dataset(
        generate_instances(20, unit_count=4, regime="in_distribution", seed=1000)
    )
    proxy, _ = train_proxy(
        train,
        validation,
        config=TrainingConfig(hidden_dim=12, epochs=8, batch_size=30, patience=4, seed=2),
    )
    return proxy


def test_benchmark_audits_all_methods_and_writes_reports(tmp_path: Path) -> None:
    proxy = _proxy_for_benchmark()
    instances = generate_instances(5, unit_count=4, regime="cost_shift", seed=4000)
    report = run_benchmark(instances, proxy, relative_gap_tolerance=0.03)
    assert len(report.rows) == 5 * 6
    assert report.summary["certified_hybrid"]["feasibility_rate"] == 1.0
    assert all(point["guarantee_violations"] == 0.0 for point in report.certificate_curve)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json_path.read_text().startswith("{")
    assert "certified_hybrid" in csv_path.read_text()


def test_compact_research_protocol_runs_all_shift_scenarios() -> None:
    _, report = run_research_experiment(
        ResearchConfig(
            train_samples=50,
            validation_samples=12,
            evaluation_samples=2,
            unit_count=4,
            hidden_dim=10,
            epochs=4,
            batch_size=25,
            patience=3,
            seed=12,
        )
    )
    assert set(report.scenarios) == {
        "in_distribution",
        "high_demand",
        "low_demand",
        "cost_shift",
        "capacity_shift",
        "combined_shift",
    }
    assert report.seed_ranges["train"][0] == 12
