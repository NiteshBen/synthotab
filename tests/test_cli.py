"""Unit tests for SynthoTab CLI."""

import os
import pytest
import pandas as pd
from synthotab.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "SynthoTab" in captured.out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_demo(tmp_path, capsys):
    report_file = str(tmp_path / "demo_report.html")
    ret = main(["demo", "--rows", "50", "--report", report_file])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SYNTHOTAB END-TO-END DEMO" in captured.out
    assert os.path.exists(report_file)


def test_cli_fit_sample_and_evaluate(tmp_path, capsys):
    # Create sample CSV
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0] * 5,
        "cat": ["A", "B", "A", "B", "A"] * 5,
    })
    input_csv = str(tmp_path / "real.csv")
    output_csv = str(tmp_path / "synth.csv")
    df.to_csv(input_csv, index=False)

    # 1. Fit-sample
    ret = main(["fit-sample", "-i", input_csv, "-o", output_csv, "-n", "25"])
    assert ret == 0
    assert os.path.exists(output_csv)

    # 2. Evaluate
    ret_eval = main(["evaluate", "--real", input_csv, "--synthetic", output_csv])
    assert ret_eval == 0
    captured = capsys.readouterr()
    assert "Overall Fidelity Score" in captured.out

    # 3. Audit Privacy
    ret_audit = main(["audit-privacy", "--real", input_csv, "--synthetic", output_csv])
    assert ret_audit == 0
    captured2 = capsys.readouterr()
    assert "Privacy Composite Score" in captured2.out


def test_cli_fit_sample_with_report(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    in_csv = str(tmp_path / "in.csv")
    out_csv = str(tmp_path / "out.csv")
    rep_html = str(tmp_path / "rep.html")
    df.to_csv(in_csv, index=False)

    ret = main(["fit-sample", "-i", in_csv, "-o", out_csv, "-n", "10", "-e", "1.0", "-r", rep_html])
    assert ret == 0
    assert os.path.exists(out_csv)
    assert os.path.exists(rep_html)

