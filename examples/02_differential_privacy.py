"""Differential Privacy Example: Synthesizing tabular data with mathematical DP guarantees."""

from synthotab import SynthoTab, load_financial_churn


def main():
    print("=" * 65)
    print("  SynthoTab: (epsilon, delta)-Differential Privacy Demo")
    print("=" * 65)

    real_df = load_financial_churn(n_samples=500, random_state=42)
    print(f"\n[*] Loaded banking customer dataset: {real_df.shape[0]} rows x {real_df.shape[1]} columns")

    # Target differential privacy budget: epsilon = 1.0 (strict privacy)
    epsilon_target = 1.0
    delta_target = 1e-5

    print(f"\n[*] Initializing SynthoTab with epsilon={epsilon_target}, delta={delta_target}...")
    model = SynthoTab(epsilon=epsilon_target, delta=delta_target, random_state=42)
    model.fit(real_df)

    # Privacy accountant ledger
    summary = model.accountant.summary()
    print("\n[+] Differential Privacy Budget Ledger:")
    print(f"    - Target Epsilon:     {summary['target_epsilon']}")
    print(f"    - Spent Epsilon:      {summary['spent_epsilon']:.4f}")
    print(f"    - Spent Delta:        {summary['spent_delta']}")
    print(f"    - Total Operations:   {summary['total_operations']}")
    for op in summary["audit_log"]:
        print(f"      * {op['operation']}: eps={op['epsilon']:.3f}, cumulative={op['cumulative_epsilon']:.3f}")

    # Generate DP synthetic records
    synth_df = model.sample(500)
    print(f"\n[+] Generated {len(synth_df)} differentially private synthetic records.")

    # Audit privacy
    privacy = model.audit_privacy(real_df, synth_df)
    print(f"[+] Privacy Score: {privacy.privacy_score:.1f}/100")
    print(f"[+] Exact Matches: {privacy.exact_match_count}")
    print(f"[+] Nearest Neighbor Distance Ratio: {privacy.nndr_mean:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
