"""Example: Generating interactive standalone HTML benchmark report."""

from synthotab import SynthoTab, load_healthcare_patients


def main():
    print("=" * 65)
    print("  SynthoTab: Interactive HTML Quality & Privacy Report")
    print("=" * 65)

    real_df = load_healthcare_patients(n_samples=600, random_state=42)
    print(f"[*] Loaded dataset: {real_df.shape[0]} rows x {real_df.shape[1]} columns")

    model = SynthoTab(epsilon=2.0, random_state=42)
    model.fit(real_df)

    synth_df = model.sample(600)

    report_path = "synthotab_quality_report.html"
    print(f"\n[*] Generating standalone HTML report with embedded SVG distribution charts...")
    full_path = model.generate_report(real_df, synth_df, output_path=report_path)
    print(f"[+] Report generated successfully at: {full_path}")
    print("[+] Open this file in any web browser to explore distribution fidelity and privacy metrics!")
    print("=" * 65)


if __name__ == "__main__":
    main()
