from __future__ import annotations

import json
from pathlib import Path

from reliable_proxy.cli import main


def test_cli_collect_train_solve_and_benchmark(tmp_path: Path, capsys) -> None:
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    checkpoint = tmp_path / "proxy.npz"
    solve_output = tmp_path / "solve.json"
    benchmark_output = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"

    assert main([
        "collect",
        "--samples",
        "40",
        "--units",
        "4",
        "--seed",
        "10",
        "--output",
        str(train_path),
    ]) == 0
    assert main([
        "collect",
        "--samples",
        "12",
        "--units",
        "4",
        "--seed",
        "1000",
        "--output",
        str(validation_path),
    ]) == 0
    assert main([
        "train",
        str(train_path),
        str(validation_path),
        "--hidden-dim",
        "8",
        "--epochs",
        "3",
        "--batch-size",
        "20",
        "--checkpoint",
        str(checkpoint),
    ]) == 0
    assert main([
        "solve",
        "--units",
        "4",
        "--seed",
        "500",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(solve_output),
    ]) == 0
    assert main([
        "benchmark",
        "--samples",
        "2",
        "--units",
        "4",
        "--seed",
        "600",
        "--checkpoint",
        str(checkpoint),
        "--output-json",
        str(benchmark_output),
        "--output-csv",
        str(benchmark_csv),
    ]) == 0
    payload = json.loads(solve_output.read_text())
    assert payload["certificate"]["valid"] is True
    assert benchmark_output.exists() and benchmark_csv.exists()
    assert capsys.readouterr().out.strip()


def test_cli_reports_invalid_arguments_as_json(capsys) -> None:
    assert main([
        "collect",
        "--samples",
        "0",
        "--output",
        "/tmp/unused.npz",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "ValueError"
