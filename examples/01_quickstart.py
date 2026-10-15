"""Quickstart example: Generating synthetic EHR healthcare records with SynthoTab."""

import pandas as pd
from synthotab import SynthoTab, load_healthcare_patients


def main():
    print("=" * 65)
    print("  SynthoTab Quickstart: Synthetic Healthcare Data Generation")
    print("=" * 65)

    # 1. Load realistic baseline healthcare dataset
    real_df = load_healthcare_patients(n_samples=500, random_state=42)
    print(f"\n[*] Loaded baseline dataset: {real_df.shape[0]} rows x {real_df.shape[1]} columns")
    print("\nSample Real Records:")
    print(real_df[["patient_id", "age", "gender", "systolic_bp", "diagnosis", "treatment_cost"]].head())

    # 2. Initialize and fit SynthoTab
    print("\n[*] Initializing SynthoTab Gaussian Copula synthesizer...")
    model = SynthoTab(random_state=42)
    model.fit(real_df)

    # 3. Sample synthetic rows
    n_synth = 500
    print(f"\n[*] Synthesizing {n_synth} new records...")
    synth_df = model.sample(n_synth)
    print("\nSample Synthetic Records:")
    print(synth_df[["patient_id", "age", "gender", "systolic_bp", "diagnosis", "treatment_cost"]].head())

    # 4. Statistical Fidelity Evaluation
    print("\n[*] Evaluating statistical fidelity & dependency preservation...")
    fidelity = model.evaluate(real_df, synth_df)
    print(f"\n[+] Overall Data Fidelity Score:      {fidelity.overall_score:.1f}%")
    print(f"[+] Marginal Distribution Fidelity:   {fidelity.marginal_fidelity_score:.1f}%")
    print(f"[+] Correlation Matrix Fidelity:      {fidelity.correlation_fidelity_score:.1f}%")
    print(f"[+] Correlation Matrix MAE:          {fidelity.correlation_mae:.4f}")

    # 5. Privacy Audit
    print("\n[*] Running privacy leakage & memorization audit...")
    privacy = model.audit_privacy(real_df, synth_df)
    print(f"\n[+] Privacy Preservation Score:       {privacy.privacy_score:.1f}/100")
    print(f"[+] Nearest Neighbor Distance Ratio:  {privacy.nndr_mean:.3f}")
    print(f"[+] Exact Verbatim Clones:            {privacy.exact_match_count} ({privacy.exact_match_ratio*100:.2f}%)")
    print(f"[+] Safe for External Sharing:        {'YES [PASS]' if privacy.is_privacy_safe else 'NO [FAIL]'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
