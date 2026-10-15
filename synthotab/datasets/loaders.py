"""Built-in realistic benchmark datasets for SynthoTab testing and demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_healthcare_patients(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic clinical EHR patient demographic and medical records dataset.

    Features realistic multi-variable dependencies (e.g., blood pressure and cholesterol
    correlated with age and BMI, treatment costs correlated with diagnosis).
    """
    rng = np.random.default_rng(random_state)

    patient_ids = [f"PAT_{i+1:05d}" for i in range(n_samples)]
    age = rng.integers(18, 82, size=n_samples)
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.52, 0.48])

    # BMI roughly normal around 26.5 with variance
    bmi = np.round(rng.normal(loc=26.5, scale=4.5, size=n_samples) + (age - 45) * 0.05, 1)
    bmi = np.clip(bmi, 18.0, 44.0)

    # Systolic BP correlated with age and BMI
    bp_noise = rng.normal(0, 10, size=n_samples)
    systolic_bp = np.round(110.0 + (age - 30) * 0.6 + (bmi - 25) * 1.2 + bp_noise, 0)
    systolic_bp = np.clip(systolic_bp, 95.0, 190.0)

    # Cholesterol correlated with age
    chol_noise = rng.normal(0, 25, size=n_samples)
    cholesterol = np.round(170.0 + (age - 30) * 0.9 + chol_noise, 0)
    cholesterol = np.clip(cholesterol, 120.0, 340.0)

    # Glucose
    gluc_noise = rng.normal(0, 20, size=n_samples)
    glucose = np.round(90.0 + (bmi - 25) * 1.5 + gluc_noise, 1)
    glucose = np.clip(glucose, 65.0, 260.0)

    smoker = rng.choice([True, False], size=n_samples, p=[0.22, 0.78])

    # Diagnosis condition influenced by blood pressure, glucose, and age
    diagnoses = []
    costs = []
    for i in range(n_samples):
        if glucose[i] > 140.0:
            diag = "Type 2 Diabetes"
            base_cost = 4500.0
        elif systolic_bp[i] > 145.0:
            diag = "Hypertension"
            base_cost = 3200.0
        elif age[i] > 60 and cholesterol[i] > 240.0:
            diag = "Coronary Artery Disease"
            base_cost = 8500.0
        else:
            diag = "Healthy Routine"
            base_cost = 800.0

        cost = base_cost + age[i] * 40.0 + (500.0 if smoker[i] else 0.0) + rng.normal(0, 300)
        diagnoses.append(diag)
        costs.append(max(250.0, round(float(cost), 2)))

    return pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "smoker": smoker,
        "diagnosis": diagnoses,
        "treatment_cost": costs,
    })


def load_financial_churn(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic retail banking customer churn dataset."""
    rng = np.random.default_rng(random_state)

    customer_ids = [f"CUST_{i+1:06d}" for i in range(n_samples)]
    credit_score = rng.integers(380, 850, size=n_samples)
    geography = rng.choice(["France", "Germany", "Spain"], size=n_samples, p=[0.50, 0.25, 0.25])
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.46, 0.54])
    age = rng.integers(18, 76, size=n_samples)
    tenure = rng.integers(0, 11, size=n_samples)

    # Balance with structural zero-inflation (some accounts have 0 balance)
    has_balance = rng.choice([True, False], size=n_samples, p=[0.65, 0.35])
    raw_balance = rng.normal(120000, 35000, size=n_samples)
    balance = np.where(has_balance, np.clip(raw_balance, 5000, 240000), 0.0)
    balance = np.round(balance, 2)

    num_of_products = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.50, 0.45, 0.04, 0.01])
    has_cr_card = rng.choice([True, False], size=n_samples, p=[0.70, 0.30])
    is_active_member = rng.choice([True, False], size=n_samples, p=[0.52, 0.48])
    estimated_salary = np.round(rng.uniform(15000, 195000, size=n_samples), 2)

    # Exited probability influenced by age, balance, and active membership
    exited = []
    for i in range(n_samples):
        prob = 0.15 + (age[i] - 35) * 0.008 + (0.10 if not is_active_member[i] else -0.05)
        prob = float(np.clip(prob, 0.02, 0.85))
        exited.append(int(rng.random() < prob))

    return pd.DataFrame({
        "customer_id": customer_ids,
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": has_cr_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        "exited": exited,
    })
