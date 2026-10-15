"""Unit tests for HTML Report Generator."""

import os
import pytest
import numpy as np
import pandas as pd
from synthotab.synthesizers.pipeline import SynthoTab


def test_html_report_generation(tmp_path):
    df = pd.DataFrame({
        "num1": np.linspace(1, 100, 50),
        "num2": np.linspace(10, 500, 50),
        "cat": ["A", "B"] * 25,
    })

    model = SynthoTab(random_state=42)
    model.fit(df)
    synth_df = model.sample(50)

    out_file = str(tmp_path / "test_report.html")
    report_path = model.generate_report(df, synth_df, output_path=out_file)

    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert "SynthoTab" in content
    assert "Quality & Privacy Benchmark" in content
    assert "<svg" in content
    assert "num1" in content
