"""Standalone, dependency-free HTML report generator for SynthoTab benchmarks."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from synthotab.metrics.fidelity import FidelityResult
from synthotab.privacy.audit import PrivacyAuditResult


def _generate_svg_histogram(
    real_vals: np.ndarray,
    synth_vals: np.ndarray,
    col_name: str,
    width: int = 320,
    height: int = 140,
) -> str:
    """Renders a clean inline SVG comparing real vs synthetic histograms."""
    r_clean = real_vals[~np.isnan(real_vals)]
    s_clean = synth_vals[~np.isnan(synth_vals)]

    if len(r_clean) == 0 or len(s_clean) == 0:
        return "<svg width='100%' height='40'><text x='10' y='25' fill='#94a3b8'>No data</text></svg>"

    min_v = min(float(np.min(r_clean)), float(np.min(s_clean)))
    max_v = max(float(np.max(r_clean)), float(np.max(s_clean)))

    if min_v == max_v:
        max_v += 1.0

    bins = np.linspace(min_v, max_v, 16)
    r_counts, _ = np.histogram(r_clean, bins=bins, density=True)
    s_counts, _ = np.histogram(s_clean, bins=bins, density=True)

    max_dens = max(float(np.max(r_counts)), float(np.max(s_counts)), 1e-6)

    pad_left = 30
    pad_right = 10
    pad_top = 15
    pad_bottom = 25
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    n_bins = len(r_counts)
    bar_w = plot_w / n_bins

    svg_elements = [
        f"<svg viewBox='0 0 {width} {height}' class='dist-svg' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect width='{width}' height='{height}' fill='#0f172a' rx='6'/>",
        f"<line x1='{pad_left}' y1='{pad_top + plot_h}' x2='{width - pad_right}' y2='{pad_top + plot_h}' stroke='#334155' stroke-width='1'/>",
    ]

    # Real bars (Cyan, semi-transparent)
    for i in range(n_bins):
        bx = pad_left + i * bar_w
        bh = (r_counts[i] / max_dens) * plot_h
        by = pad_top + plot_h - bh
        svg_elements.append(
            f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bar_w*0.85:.1f}' height='{bh:.1f}' fill='#0ea5e9' fill-opacity='0.45' rx='2'/>"
        )

    # Synthetic line / points (Emerald)
    points = []
    for i in range(n_bins):
        px = pad_left + (i + 0.5) * bar_w
        py = pad_top + plot_h - (s_counts[i] / max_dens) * plot_h
        points.append(f"{px:.1f},{py:.1f}")

    if points:
        pts_str = " ".join(points)
        svg_elements.append(f"<polyline points='{pts_str}' fill='none' stroke='#10b981' stroke-width='2.2' stroke-linecap='round'/>")
        for pt in points:
            x, y = pt.split(",")
            svg_elements.append(f"<circle cx='{x}' cy='{y}' r='2.5' fill='#10b981'/>")

    # Min / Max axis labels
    svg_elements.append(f"<text x='{pad_left}' y='{height - 8}' font-size='10' fill='#94a3b8' font-family='sans-serif'>{min_v:.1f}</text>")
    svg_elements.append(f"<text x='{width - pad_right - 25}' y='{height - 8}' font-size='10' fill='#94a3b8' font-family='sans-serif'>{max_v:.1f}</text>")
    svg_elements.append("</svg>")
    return "".join(svg_elements)


def generate_html_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    fidelity: FidelityResult,
    privacy: PrivacyAuditResult,
    privacy_summary: Optional[Dict[str, Any]] = None,
    output_path: str = "synthotab_report.html",
    title: str = "SynthoTab Synthetic Data Quality & Privacy Report",
) -> str:
    """Constructs and saves the interactive HTML benchmark report."""
    num_cols = [c for c, f in fidelity.column_fidelities.items() if f.column_type == "numerical"][:6]

    chart_cards = []
    for col in num_cols:
        r_vals = real_df[col].dropna().to_numpy(dtype=np.float64)
        s_vals = synthetic_df[col].dropna().to_numpy(dtype=np.float64)
        cf = fidelity.column_fidelities[col]
        svg_chart = _generate_svg_histogram(r_vals, s_vals, col)

        card_html = f"""
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-col-name">{col}</span>
                <span class="chart-badge">KS Score: {cf.fidelity_score*100:.1f}%</span>
            </div>
            <div class="chart-body">
                {svg_chart}
            </div>
            <div class="chart-footer">
                <span>Real μ: {cf.real_mean:.2f}</span>
                <span>Synth μ: {cf.synth_mean:.2f}</span>
            </div>
        </div>
        """
        chart_cards.append(card_html)

    # Column breakdown table rows
    table_rows = []
    for col, cf in fidelity.column_fidelities.items():
        score_pct = cf.fidelity_score * 100
        score_color = "#10b981" if score_pct >= 85 else "#f59e0b" if score_pct >= 70 else "#ef4444"
        r_mean_str = f"{cf.real_mean:.2f}" if cf.real_mean is not None else "-"
        s_mean_str = f"{cf.synth_mean:.2f}" if cf.synth_mean is not None else "-"
        p_val_str = f"{cf.p_value:.4f}" if cf.p_value is not None else "-"

        table_rows.append(f"""
        <tr>
            <td class="col-name-cell">{cf.name}</td>
            <td><span class="type-pill {cf.column_type}">{cf.column_type}</span></td>
            <td>{cf.metric_name}</td>
            <td>{cf.metric_value:.4f}</td>
            <td>{p_val_str}</td>
            <td>{r_mean_str}</td>
            <td>{s_mean_str}</td>
            <td>
                <div class="score-bar-wrap">
                    <div class="score-bar" style="width: {score_pct:.1f}%; background: {score_color};"></div>
                    <span class="score-text">{score_pct:.1f}%</span>
                </div>
            </td>
        </tr>
        """)

    charts_section = "".join(chart_cards)
    rows_section = "".join(table_rows)

    dp_spent = f"{privacy_summary['spent_epsilon']:.2f}" if privacy_summary else "N/A"
    dp_target = f"{privacy_summary['target_epsilon']:.2f}" if privacy_summary and privacy_summary.get("target_epsilon", -1) > 0 else "Unbounded"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-card: #1e293b;
            --border: #334155;
            --primary: #0ea5e9;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .brand-title {{ font-size: 26px; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 10px; }}
        .brand-title span {{ color: var(--primary); }}
        .brand-subtitle {{ font-size: 14px; color: var(--text-muted); margin-top: 4px; }}
        .badge-bar {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .badge {{
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--primary);
        }}
        .badge.success {{ color: var(--success); border-color: rgba(16, 185, 129, 0.4); }}
        .badge.purple {{ color: #c084fc; border-color: rgba(139, 92, 246, 0.4); }}

        /* Metric Grid */
        .grid-kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px 24px;
            transition: transform 0.2s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); }}
        .kpi-label {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; }}
        .kpi-val {{ font-size: 32px; font-weight: 800; color: var(--text-main); margin: 6px 0; }}
        .kpi-val.cyan {{ color: var(--primary); }}
        .kpi-val.emerald {{ color: var(--success); }}
        .kpi-val.purple {{ color: #c084fc; }}
        .kpi-desc {{ font-size: 13px; color: var(--text-muted); }}

        /* Distribution Charts Grid */
        .section-header {{ margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }}
        .section-title {{ font-size: 20px; font-weight: 700; color: var(--text-main); }}
        .legend {{ font-size: 13px; display: flex; gap: 16px; align-items: center; color: var(--text-muted); }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        .chart-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
        }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .chart-col-name {{ font-weight: 600; font-size: 15px; color: var(--text-main); }}
        .chart-badge {{ font-size: 12px; font-weight: 600; color: var(--success); background: rgba(16, 185, 129, 0.15); padding: 3px 8px; border-radius: 6px; }}
        .chart-footer {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); }}
        .dist-svg {{ width: 100%; height: auto; display: block; }}

        /* Table */
        .table-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            margin-bottom: 32px;
        }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th {{ background: #1e293b; color: var(--text-muted); font-weight: 600; padding: 14px 18px; border-bottom: 1px solid var(--border); text-transform: uppercase; font-size: 12px; letter-spacing: 0.04em; }}
        td {{ padding: 14px 18px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1; }}
        tr:hover td {{ background: rgba(255,255,255,0.02); }}
        .col-name-cell {{ font-weight: 600; color: var(--text-main); }}
        .type-pill {{ font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; }}
        .type-pill.numerical {{ background: rgba(14, 165, 233, 0.15); color: #38bdf8; }}
        .type-pill.categorical {{ background: rgba(139, 92, 246, 0.15); color: #c084fc; }}
        .score-bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
        .score-bar {{ height: 8px; border-radius: 4px; min-width: 4px; }}
        .score-text {{ font-size: 12px; font-weight: 600; color: var(--text-main); width: 44px; }}

        footer {{
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <div class="brand-title"><span>SynthoTab</span> Quality & Privacy Benchmark</div>
                <div class="brand-subtitle">Automated Statistical Fidelity, Gaussian Copula Dependencies & Differential Privacy Audit</div>
            </div>
            <div class="badge-bar">
                <span class="badge success">Fidelity: {fidelity.overall_score:.1f}%</span>
                <span class="badge purple">Privacy: {privacy.privacy_score:.1f}/100</span>
                <span class="badge">DP Epsilon: {dp_spent}</span>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="grid-kpi">
            <div class="kpi-card">
                <div class="kpi-label">Overall Data Fidelity</div>
                <div class="kpi-val cyan">{fidelity.overall_score:.1f}%</div>
                <div class="kpi-desc">Weighted marginal (55%) + correlation (45%) score</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Correlation Error (MAE)</div>
                <div class="kpi-val emerald">{fidelity.correlation_mae:.3f}</div>
                <div class="kpi-desc">Mean absolute error across Pearson correlation coefficients</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Privacy Risk Score</div>
                <div class="kpi-val purple">{privacy.privacy_score:.1f}<span style="font-size: 18px; font-weight: 500;">/100</span></div>
                <div class="kpi-desc">{privacy.summary}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Nearest Neighbor Ratio (NNDR)</div>
                <div class="kpi-val emerald">{privacy.nndr_mean:.3f}</div>
                <div class="kpi-desc">d1/d2 ratio (>0.70 confirms zero record memorization)</div>
            </div>
        </div>

        <!-- Overlaid Distribution Visualizations -->
        <div class="section-header">
            <div class="section-title">Marginal Distribution Overlays</div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #0ea5e9;"></div>Real Data</div>
                <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div>Synthetic (SynthoTab)</div>
            </div>
        </div>
        <div class="charts-grid">
            {charts_section}
        </div>

        <!-- Comprehensive Column Fidelity Table -->
        <div class="section-header">
            <div class="section-title">Column-by-Column Statistical Breakdown</div>
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Benchmark Metric</th>
                        <th>Stat Value</th>
                        <th>p-value</th>
                        <th>Real Mean</th>
                        <th>Synth Mean</th>
                        <th>Fidelity</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_section}
                </tbody>
            </table>
        </div>

        <footer>
            Generated automatically by <strong>SynthoTab</strong> • Open Source Python Framework • Developed by Nitesh Beniwal
        </footer>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)
