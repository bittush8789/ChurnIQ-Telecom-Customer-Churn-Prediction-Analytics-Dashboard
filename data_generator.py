"""
Synthetic Telecom Customer Churn Dataset Generator
===================================================

Every record produced by this module is **synthetic** (randomly generated).
The relationships between customer behaviour and churn are hand-crafted so the
dataset behaves in a realistic way for a machine-learning demo.  It does NOT
represent real customers, real telecom operators, or real business results.

Usage
-----
    python data_generator.py

or from Python::

    from data_generator import generate_telecom_data
    df = generate_telecom_data(n_samples=7000, random_state=42)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES = 7_000
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "telecom_churn_synthetic.csv"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_telecom_data(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a synthetic telecom customer dataset.

    Parameters
    ----------
    n_samples:
        Number of customer records to generate (default 7,000).
    random_state:
        Seed used by NumPy so the dataset is fully reproducible.

    Returns
    -------
    pandas.DataFrame
        One row per customer with the requested telecom features.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 to create a useful dataset.")

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.495, 0.505])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])

    # ------------------------------------------------------------------
    # Contract type and tenure (longer contracts -> longer tenure)
    # ------------------------------------------------------------------
    contract_type = rng.choice(
        CONTRACTS, size=n_samples, p=[0.55, 0.24, 0.21]
    )
    tenure = np.empty(n_samples, dtype=int)
    tenure_params = {
        "Month-to-month": (1.2, 2.0),
        "One year": (1.8, 1.5),
        "Two year": (2.4, 1.2),
    }
    for contract_name, (alpha, beta) in tenure_params.items():
        mask = contract_type == contract_name
        raw = rng.beta(alpha, beta, size=int(mask.sum())) * 72
        tenure[mask] = np.clip(np.round(raw), 0, 72).astype(int)

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    internet_service = rng.choice(
        INTERNET_SERVICES, size=n_samples, p=[0.34, 0.44, 0.22]
    )
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])

    has_internet = internet_service != "No"
    has_phone = phone_service == "Yes"

    multiple_lines = np.where(
        has_phone,
        rng.choice(["Yes", "No"], size=n_samples, p=[0.46, 0.54]),
        "No phone service",
    )

    online_security = np.where(
        has_internet,
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]),
        "No internet service",
    )
    tech_support = np.where(
        has_internet,
        rng.choice(["Yes", "No"], size=n_samples, p=[0.43, 0.57]),
        "No internet service",
    )

    payment_method = rng.choice(
        PAYMENT_METHODS, size=n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])

    # ------------------------------------------------------------------
    # Usage and billing
    # ------------------------------------------------------------------
    monthly_usage = np.empty(n_samples, dtype=float)
    usage_params = {
        "No": (0.0, 22.0),       # very light usage without home internet
        "DSL": (4.0, 22.0),      # gamma(shape, scale) -> mean ~88 GB
        "Fiber optic": (4.0, 58.0),  # mean ~232 GB
    }
    for service, (shape, scale) in usage_params.items():
        mask = internet_service == service
        if service == "No":
            monthly_usage[mask] = rng.uniform(0, scale, size=int(mask.sum()))
        else:
            monthly_usage[mask] = rng.gamma(shape, scale, size=int(mask.sum()))
    monthly_usage = np.round(np.clip(monthly_usage, 0, 900), 1)

    # Monthly charges are driven by the services the customer subscribes to.
    monthly_charges = (
        18.0
        + np.where(has_phone, 12.0, 0.0)
        + np.select(
            [internet_service == "DSL", internet_service == "Fiber optic"],
            [24.0, 52.0],
            default=0.0,
        )
        + np.where(multiple_lines == "Yes", 8.0, 0.0)
        + np.where(online_security == "Yes", 5.0, 0.0)
        + np.where(tech_support == "Yes", 5.0, 0.0)
        + rng.normal(0.0, 4.0, size=n_samples)
    )
    monthly_charges = np.round(np.clip(monthly_charges, 18.0, 135.0), 2)

    total_charges = np.round(
        np.maximum(
            tenure * monthly_charges * rng.normal(1.0, 0.035, size=n_samples),
            0.0,
        ),
        2,
    )

    # Support calls are more common for expensive plans, new customers and fiber.
    call_lambda = (
        1.0
        + monthly_charges / 70.0
        + np.where(tenure < 12, 1.0, 0.0)
        + np.where(internet_service == "Fiber optic", 0.4, 0.0)
    )
    support_calls = np.clip(
        rng.poisson(np.clip(call_lambda, 0.3, 7.0)), 0, 12
    ).astype(int)

    # ------------------------------------------------------------------
    # Churn propensity
    # ------------------------------------------------------------------
    # The log-odds below encode realistic churn drivers.  Random noise keeps
    # the problem non-deterministic, so a model cannot achieve perfect scores.
    logit = (
        -1.68
        + np.select(
            [contract_type == "Month-to-month", contract_type == "One year"],
            [1.15, 0.15],
            default=-0.75,
        )
        + np.select(
            [internet_service == "Fiber optic", internet_service == "DSL"],
            [0.55, -0.05],
            default=-0.25,
        )
        + np.where(online_security == "No", 0.45, 0.0)
        + np.where(tech_support == "No", 0.40, 0.0)
        + np.where(payment_method == "Electronic check", 0.45, 0.0)
        + np.where(paperless_billing == "Yes", 0.18, 0.0)
        + senior_citizen * 0.22
        + (monthly_charges - 65.0) * 0.012
        + support_calls * 0.16
        + (monthly_usage - 120.0) * 0.0015
        + np.where(multiple_lines == "Yes", 0.08, 0.0)
        - tenure * 0.045
        + np.where(phone_service == "No", -0.15, 0.0)
        + rng.normal(0.0, 0.45, size=n_samples)
    )
    churn_probability = _sigmoid(logit)
    churn_flag = rng.binomial(1, churn_probability)

    # ------------------------------------------------------------------
    # Assemble the dataset
    # ------------------------------------------------------------------
    customer_ids = [f"TC-{i:06d}" for i in range(1, n_samples + 1)]
    df = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Tenure": tenure,
            "ContractType": contract_type,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "InternetService": internet_service,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing,
            "MonthlyUsage": monthly_usage,
            "CustomerSupportCalls": support_calls,
            "Churn": np.where(churn_flag == 1, "Yes", "No"),
        }
    )

    # ------------------------------------------------------------------
    # Deliberately add a small amount of realistic missing data so the
    # machine-learning pipeline has to demonstrate imputation.
    # ------------------------------------------------------------------
    missing_plan = {
        "TotalCharges": 0.012,
        "MonthlyUsage": 0.006,
        "PaymentMethod": 0.002,
        "CustomerSupportCalls": 0.004,
    }
    for column, rate in missing_plan.items():
        mask = rng.random(n_samples) < rate
        df.loc[mask, column] = np.nan

    # Support calls are conceptually integers; keep them nullable-friendly.
    df["CustomerSupportCalls"] = df["CustomerSupportCalls"].astype("float64")

    return df


def save_dataset(
    df: pd.DataFrame,
    path: str | Path = DATA_PATH,
) -> Path:
    """Save the synthetic dataset as CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    """Generate the dataset, save it, and print a short summary."""
    df = generate_telecom_data()
    path = save_dataset(df)

    churn_rate = (df["Churn"] == "Yes").mean()
    print("Synthetic telecom dataset generated (NOT real customer data).")
    print(f"Rows            : {len(df):,}")
    print(f"Columns         : {len(df.columns)}")
    print(f"Churn rate      : {churn_rate:.1%}")
    print(f"Missing values  : {int(df.isna().sum().sum()):,}")
    print(f"Saved to        : {path}")


if __name__ == "__main__":
    main()
