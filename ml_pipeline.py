"""
Machine-Learning Pipeline for Telecom Customer Churn
====================================================

This module contains every machine-learning step:

1. Load and clean the synthetic dataset.
2. Build a leakage-safe preprocessing pipeline (imputation + encoding + scaling).
3. Train and compare four classifiers with 5-fold stratified cross-validation.
4. Select the best model by mean validation F1-score.
5. Evaluate the selected model once on the untouched test set.
6. Persist the fitted pipeline with Joblib and expose prediction helpers.

All preprocessing lives *inside* the scikit-learn pipeline, so it is fitted
only on training folds and never sees validation/test data.

Run directly to (re)build the dataset and model::

    python ml_pipeline.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # XGBoost is optional so the project still runs if it is unavailable.
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False

from data_generator import DATA_PATH, generate_telecom_data

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyUsage",
    "CustomerSupportCalls",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "ContractType",
    "InternetService",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Probability cut-offs used to translate a model score into a business risk band.
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30
PREDICTION_THRESHOLD = 0.50

# Basic input validation ranges for the prediction form.
NUMERIC_BOUNDS = {
    "SeniorCitizen": (0, 1),
    "Tenure": (0, 72),
    "MonthlyCharges": (0.0, 300.0),
    "TotalCharges": (0.0, 50_000.0),
    "MonthlyUsage": (0.0, 2_000.0),
    "CustomerSupportCalls": (0, 50),
}


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light, deterministic cleaning to the raw synthetic dataset."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN])
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # TotalCharges is sometimes blank in telecom exports; force numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Keep one row per customer and drop records without a target label.
    df = df.drop_duplicates(subset=[ID_COLUMN]).copy()
    df = df[df[TARGET_COLUMN].isin(["Yes", "No"])].copy()

    if len(df) < 200:
        raise ValueError("Not enough valid rows to train a reliable model.")

    return df.reset_index(drop=True)


def ensure_dataset(
    force: bool = False,
    n_samples: int = 7_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load the synthetic dataset, generating it on first use if necessary."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if force or not DATA_PATH.exists():
        raw = generate_telecom_data(n_samples=n_samples, random_state=random_state)
        raw.to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    return clean_data(df)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into feature matrix X and binary target y."""
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing and model definitions
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Create a fresh preprocessing transformer.

    A new instance is created for every model/fold so that imputers, scalers
    and encoders are fitted only on the data they are allowed to see.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_pipeline(model: Any) -> Pipeline:
    """Wrap a classifier in a preprocessing pipeline (no data leakage)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_models() -> dict[str, Any]:
    """Return the four candidate classifiers with reproducible settings."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=2_000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=40,
            min_samples_split=20,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=8,
            min_samples_split=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=280,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return models


# ---------------------------------------------------------------------------
# Cross-validation and evaluation
# ---------------------------------------------------------------------------
def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """Compare all candidate models using stratified k-fold cross-validation."""
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: list[dict[str, Any]] = []
    for name, model in build_models().items():
        pipeline = build_pipeline(model)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row: dict[str, Any] = {
            "Model": name,
            "CV F1": float(np.mean(scores["test_f1"])),
            "CV F1 Std": float(np.std(scores["test_f1"])),
            "CV Accuracy": float(np.mean(scores["test_accuracy"])),
            "CV Precision": float(np.mean(scores["test_precision"])),
            "CV Recall": float(np.mean(scores["test_recall"])),
            "CV ROC-AUC": float(np.mean(scores["test_roc_auc"])),
        }
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=["CV F1", "CV ROC-AUC"], ascending=False
    ).reset_index(drop=True)
    return comparison


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a fitted pipeline on the untouched test set."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "test_rows": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
    }


def _pretty_feature_name(name: str) -> str:
    """Turn sklearn transformer names into readable dashboard labels."""
    name = name.split("__", 1)[-1]
    for column in CATEGORICAL_FEATURES:
        prefix = f"{column}_"
        if name.startswith(prefix):
            return f"{column}: {name[len(prefix):].replace('_', ' ')}"
    return name.replace("_", " ")


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importances (or absolute coefficients) for the model."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = pd.DataFrame(
        {
            "Feature": [_pretty_feature_name(name) for name in feature_names],
            "Importance": values,
        }
    )
    importance = importance.sort_values("Importance", ascending=False)
    return importance.head(20).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training orchestration and persistence
# ---------------------------------------------------------------------------
def train_models(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Run the full training workflow and return a serialisable model bundle."""
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    comparison = cross_validate_models(X_train, y_train, cv_folds=cv_folds)
    best_model_name = str(comparison.iloc[0]["Model"])

    # Refit the winning configuration on the complete training set.
    best_model = build_models()[best_model_name]
    final_pipeline = build_pipeline(best_model)
    final_pipeline.fit(X_train, y_train)

    test_results = evaluate_model(final_pipeline, X_test, y_test)
    feature_importance = extract_feature_importance(final_pipeline)

    bundle: dict[str, Any] = {
        "pipeline": final_pipeline,
        "best_model_name": best_model_name,
        "comparison": comparison,
        "test_metrics": {
            key: value
            for key, value in test_results.items()
            if key not in {"confusion_matrix", "roc_curve"}
        },
        "confusion_matrix": test_results["confusion_matrix"],
        "roc_curve": test_results["roc_curve"],
        "feature_importance": feature_importance,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": int(cv_folds),
        "random_state": RANDOM_STATE,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "dataset_rows": int(len(df)),
        "dataset_churn_rate": float(y.mean()),
        "data_source": "Synthetic (generated for demonstration only)",
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    return bundle


def save_model(bundle: dict[str, Any], path: Path = MODEL_PATH) -> Path:
    """Persist the fitted model bundle with Joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def save_metrics_summary(bundle: dict[str, Any], path: Path = METRICS_PATH) -> Path:
    """Save a small human-readable JSON summary next to the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_source": bundle["data_source"],
        "best_model": bundle["best_model_name"],
        "trained_at": bundle["trained_at"],
        "random_state": bundle["random_state"],
        "cv_folds": bundle["cv_folds"],
        "train_rows": bundle["train_rows"],
        "test_rows": bundle["test_rows"],
        "dataset_rows": bundle["dataset_rows"],
        "dataset_churn_rate": round(bundle["dataset_churn_rate"], 4),
        "test_metrics": {
            key: round(float(value), 4)
            for key, value in bundle["test_metrics"].items()
        },
        "cv_results": bundle["comparison"].round(4).to_dict(orient="records"),
        "note": (
            "Metrics come from synthetic data and are for demonstration only. "
            "They do not represent real telecom business performance."
        ),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load a previously saved model bundle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("The saved model file is not a valid model bundle.")
    return bundle


def ensure_model(
    force: bool = False,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Load the trained model, training a fresh one when needed."""
    if MODEL_PATH.exists() and not force:
        try:
            return load_model()
        except Exception:
            # A corrupt/incompatible artifact should never block the dashboard.
            pass

    if df is None:
        df = ensure_dataset()

    bundle = train_models(df)
    save_model(bundle)
    save_metrics_summary(bundle)
    return bundle


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def risk_level(probability: float) -> str:
    """Map a churn probability to a simple business risk band."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def validate_customer(customer: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a single customer payload.

    Raises
    ------
    ValueError
        If a required field is missing or a numeric value is out of range.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in customer]
    if missing:
        raise ValueError(f"Missing customer fields: {missing}")

    clean: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        value = customer[column]

        if column in NUMERIC_FEATURES:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must be a number.") from exc
            if not np.isfinite(numeric_value):
                raise ValueError(f"{column} must be a finite number.")
            low, high = NUMERIC_BOUNDS[column]
            if not low <= numeric_value <= high:
                raise ValueError(
                    f"{column} must be between {low} and {high} "
                    f"(received {numeric_value})."
                )
            clean[column] = numeric_value
        else:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise ValueError(f"{column} cannot be empty.")
            clean[column] = str(value).strip()

    return clean


def predict_customer(
    bundle: dict[str, Any],
    customer: dict[str, Any],
) -> dict[str, Any]:
    """Predict churn probability and risk level for one customer."""
    clean_customer = validate_customer(customer)
    X = pd.DataFrame([clean_customer], columns=FEATURE_COLUMNS)

    pipeline: Pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(X)[0, 1])
    threshold = float(bundle.get("prediction_threshold", PREDICTION_THRESHOLD))

    return {
        "churn_probability": probability,
        "prediction": "Yes" if probability >= threshold else "No",
        "risk_level": risk_level(probability),
        "threshold": threshold,
        "model_name": bundle.get("best_model_name", "Unknown"),
    }


def score_dataframe(
    bundle: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every customer in a dataframe (batch scoring for the dashboard)."""
    scored = df.copy()
    X = scored[FEATURE_COLUMNS].copy()
    probabilities = bundle["pipeline"].predict_proba(X)[:, 1]

    scored["ChurnProbability"] = probabilities
    scored["PredictedChurn"] = np.where(
        probabilities >= float(bundle.get("prediction_threshold", 0.5)),
        "Yes",
        "No",
    )
    scored["RiskLevel"] = [risk_level(float(p)) for p in probabilities]
    return scored


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate data if needed, train all models, and print a summary."""
    df = ensure_dataset()
    bundle = train_models(df)
    save_model(bundle)
    save_metrics_summary(bundle)

    print("=" * 68)
    print("TELECOM CHURN MODEL TRAINING COMPLETE (synthetic data)")
    print("=" * 68)
    print(f"Rows: {len(df):,} | Train: {bundle['train_rows']:,} | "
          f"Test: {bundle['test_rows']:,}")
    print(f"Churn rate: {bundle['dataset_churn_rate']:.1%}")
    print(f"Cross-validation: {bundle['cv_folds']}-fold stratified\n")
    print(bundle["comparison"].to_string(index=False))
    print(f"\nSelected model: {bundle['best_model_name']}")
    print("Untouched test-set metrics:")
    for metric, value in bundle["test_metrics"].items():
        print(f"  {metric:<10}: {value:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print("NOTE: Synthetic data - results are for demonstration only.")


if __name__ == "__main__":
    main()
