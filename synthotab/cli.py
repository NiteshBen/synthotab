"""Command Line Interface (CLI) for SynthoTab."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
import pandas as pd

from synthotab.synthesizers.pipeline import SynthoTab
from synthotab.metrics.fidelity import FidelityEvaluator
from synthotab.privacy.audit import PrivacyAuditor
from synthotab.datasets.loaders import load_healthcare_patients, load_financial_churn


def create_parser() -> argparse.ArgumentParser:
    """Creates the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="synthotab",
        description="SynthoTab: Production-grade synthetic tabular data generation & differential privacy engine.",
    )
    parser.add_argument("--version", action="version", version="synthotab 0.1.0")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. fit-sample
    p_fit = subparsers.add_parser("fit-sample", help="Fit synthesizer on input CSV and generate synthetic data")
    p_fit.add_argument("--input", "-i", required=True, help="Path to input real CSV file")
    p_fit.add_argument("--output", "-o", default="synthetic_output.csv", help="Path to write synthetic CSV file")
    p_fit.add_argument("--rows", "-n", type=int, default=None, help="Number of synthetic rows to generate")
    p_fit.add_argument("--epsilon", "-e", type=float, default=None, help="Differential privacy epsilon budget")
    p_fit.add_argument("--report", "-r", type=str, default=None, help="Optional HTML report output path")

    # 2. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate statistical fidelity between real and synthetic CSVs")
    p_eval.add_argument("--real", required=True, help="Path to real baseline CSV file")
    p_eval.add_argument("--synthetic", required=True, help="Path to synthetic CSV file")
    p_eval.add_argument("--report", type=str, default=None, help="Optional HTML report output path")

    # 3. audit-privacy
    p_audit = subparsers.add_parser("audit-privacy", help="Audit synthetic CSV against real CSV for privacy leakage")
    p_audit.add_argument("--real", required=True, help="Path to real baseline CSV file")
    p_audit.add_argument("--synthetic", required=True, help="Path to synthetic CSV file")

    # 4. demo
    p_demo = subparsers.add_parser("demo", help="Run self-contained demonstration with built-in healthcare dataset")
    p_demo.add_argument("--dataset", choices=["healthcare", "churn"], default="healthcare", help="Demo dataset to use")
    p_demo.add_argument("--rows", type=int, default=300, help="Number of rows to generate")
    p_demo.add_argument("--epsilon", type=float, default=1.5, help="Differential privacy epsilon")
    p_demo.add_argument("--report", type=str, default="demo_report.html", help="Path for generated HTML report")

    return parser


def run_fit_sample(args: argparse.Namespace) -> int:
    """Executes fit-sample subcommand."""
    print(f"[*] Loading input data from: {args.input}")
    try:
        real_df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[!] Error reading input CSV: {e}", file=sys.stderr)
        return 1

    n_rows = args.rows if args.rows is not None else len(real_df)
    print(f"[*] Input shape: {real_df.shape[0]} rows, {real_df.shape[1]} columns")
    if args.epsilon is not None:
        print(f"[*] Differential Privacy active with epsilon = {args.epsilon:.2f}")

    synthesizer = SynthoTab(epsilon=args.epsilon, random_state=42)
    print("[*] Fitting Gaussian Copula & Categorical dependencies...")
    synthesizer.fit(real_df)

    print(f"[*] Sampling {n_rows} synthetic records...")
    synth_df = synthesizer.sample(n_rows)
    synth_df.to_csv(args.output, index=False)
    print(f"[+] Successfully saved synthetic data to: {args.output}")

    if args.report:
        print(f"[*] Generating quality & privacy report at: {args.report}...")
        report_path = synthesizer.generate_report(real_df, synth_df, output_path=args.report)
        print(f"[+] Report generated at: {report_path}")

    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Executes evaluate subcommand."""
    try:
        real_df = pd.read_csv(args.real)
        synth_df = pd.read_csv(args.synthetic)
    except Exception as e:
        print(f"[!] Error reading datasets: {e}", file=sys.stderr)
        return 1

    evaluator = FidelityEvaluator()
    res = evaluator.evaluate(real_df, synth_df)
    print("=" * 60)
    print("  SYNTHOTAB STATISTICAL FIDELITY EVALUATION")
    print("=" * 60)
    print(f"Overall Fidelity Score:      {res.overall_score:.1f}%")
    print(f"Marginal Distribution Score: {res.marginal_fidelity_score:.1f}%")
    print(f"Correlation Fidelity Score:  {res.correlation_fidelity_score:.1f}%")
    print(f"Correlation Matrix MAE:      {res.correlation_mae:.4f}")
    print("=" * 60)

    if args.report:
        from synthotab.reporting.html_report import generate_html_report
        auditor = PrivacyAuditor()
        priv_res = auditor.audit(real_df, synth_df)
        generate_html_report(real_df, synth_df, res, priv_res, output_path=args.report)
        print(f"[+] Generated benchmark report at: {args.report}")

    return 0


def run_audit_privacy(args: argparse.Namespace) -> int:
    """Executes audit-privacy subcommand."""
    try:
        real_df = pd.read_csv(args.real)
        synth_df = pd.read_csv(args.synthetic)
    except Exception as e:
        print(f"[!] Error reading datasets: {e}", file=sys.stderr)
        return 1

    auditor = PrivacyAuditor()
    res = auditor.audit(real_df, synth_df)
    print("=" * 60)
    print("  SYNTHOTAB PRIVACY & LEAKAGE AUDIT")
    print("=" * 60)
    print(f"Privacy Composite Score:     {res.privacy_score:.1f} / 100")
    print(f"Exact Match Count:           {res.exact_match_count} ({res.exact_match_ratio*100:.3f}%)")
    print(f"Nearest Neighbor Ratio NNDR: {res.nndr_mean:.3f}")
    print(f"Membership Inference Risk:   {res.membership_inference_risk:.3f}")
    print(f"Safe for Release:            {'YES [PASS]' if res.is_privacy_safe else 'NO [WARN]'}")
    print(f"Summary:                     {res.summary}")
    print("=" * 60)
    return 0


def run_demo(args: argparse.Namespace) -> int:
    """Executes built-in demonstration."""
    print("=" * 65)
    print("  SYNTHOTAB END-TO-END DEMO")
    print("=" * 65)

    if args.dataset == "healthcare":
        print("[*] Generating realistic Electronic Health Records (EHR) dataset...")
        real_df = load_healthcare_patients(n_samples=args.rows, random_state=42)
    else:
        print("[*] Generating realistic Bank Customer Churn dataset...")
        real_df = load_financial_churn(n_samples=args.rows, random_state=42)

    print(f"[*] Real Data: {real_df.shape[0]} rows x {real_df.shape[1]} columns")
    print("[*] Fitting SynthoTab with Differential Privacy (epsilon = 1.5)...")
    model = SynthoTab(epsilon=args.epsilon, random_state=42)
    model.fit(real_df)

    print(f"[*] Synthesizing {args.rows} synthetic records...")
    synth_df = model.sample(args.rows)

    print("[*] Evaluating statistical fidelity & running privacy audit...")
    fidelity = model.evaluate(real_df, synth_df)
    privacy = model.audit_privacy(real_df, synth_df)

    print(f"[+] Fidelity Score: {fidelity.overall_score:.1f}%")
    print(f"[+] Privacy Score:  {privacy.privacy_score:.1f}/100")
    print(f"[+] NNDR Ratio:     {privacy.nndr_mean:.3f} (Zero memorization)")
    print(f"[+] Exact Matches:  {privacy.exact_match_count}")

    if args.report:
        report_path = model.generate_report(real_df, synth_df, output_path=args.report)
        print(f"[+] Standalone interactive report written to: {report_path}")

    print("=" * 65)
    print("[+] Demo completed successfully!")
    print("=" * 65)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "fit-sample":
        return run_fit_sample(args)
    elif args.subcommand == "evaluate":
        return run_evaluate(args)
    elif args.subcommand == "audit-privacy":
        return run_audit_privacy(args)
    elif args.subcommand == "demo":
        return run_demo(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
