"""
model/train.py
Model training and SHAP explainability engine for SHAP-CC / Shera-Real.
Trains Gradient Boosted Trees (XGBoost) to predict node placement scores,
energy proxy impact, and contention risk.
Computes and visualizes SHAP explanations.
"""

import argparse
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "cpu_req_cores",
    "mem_req_mb",
    "net_mbps",
    "duration_sec",
    "workload_type_cpu",
    "workload_type_memory",
    "workload_type_network",
    "workload_type_mixed",
    "node_cpu_util_pct",
    "node_mem_util_pct",
    "node_active_pods",
    "node_power_efficiency",
    "post_alloc_cpu_pct",
    "post_alloc_mem_pct",
    "resource_imbalance",
]

TARGET_COLUMNS = {
    "placement_score": "node_placement_score",
    "energy_proxy_joules": "energy_proxy_joules",
    "contention_risk": "contention_risk",
}

def train_and_evaluate(
    dataset_path: str = "data/workload_features_dataset.csv",
    output_dir: str = "model",
    test_size: float = 0.2,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading training data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Validate columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    X = df[FEATURE_COLUMNS].copy()
    y_score = df["placement_score"].copy()
    y_energy = df["energy_proxy_joules"].copy()
    y_contention = df["contention_risk"].copy()

    X_train, X_test, y_s_train, y_s_test = train_test_split(
        X, y_score, test_size=test_size, random_state=random_state
    )
    _, _, y_e_train, y_e_test = train_test_split(
        X, y_energy, test_size=test_size, random_state=random_state
    )
    _, _, y_c_train, y_c_test = train_test_split(
        X, y_contention, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 1. Train Primary Placement Scoring Model
    print("Training XGBoost Placement Score Model...")
    model_score = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
    )
    model_score.fit(X_train, y_s_train)

    preds_s = model_score.predict(X_test)
    r2_s = r2_score(y_s_test, preds_s)
    mae_s = mean_absolute_error(y_s_test, preds_s)
    rmse_s = np.sqrt(mean_squared_error(y_s_test, preds_s))
    print(f"[Placement Score Model] R²: {r2_s:.4f} | MAE: {mae_s:.3f} | RMSE: {rmse_s:.3f}")

    # 2. Train Energy Proxy Model
    print("Training Energy Proxy Model...")
    model_energy = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    model_energy.fit(X_train, y_e_train)
    preds_e = model_energy.predict(X_test)
    r2_e = r2_score(y_e_test, preds_e)
    print(f"[Energy Model] R²: {r2_e:.4f} | MAE: {mean_absolute_error(y_e_test, preds_e):.2f} Joules")

    # 3. Train Contention Risk Model
    print("Training Contention Risk Model...")
    model_contention = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    model_contention.fit(X_train, y_c_train)
    preds_c = model_contention.predict(X_test)
    r2_c = r2_score(y_c_test, preds_c)
    print(f"[Contention Model] R²: {r2_c:.4f} | MAE: {mean_absolute_error(y_c_test, preds_c):.4f}")

    # 4. Compute SHAP Values on Placement Model
    print("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model_score)
    sample_size = min(500, len(X_test))
    X_shap_sample = X_test.iloc[:sample_size]
    shap_values = explainer(X_shap_sample)

    # Calculate mean absolute SHAP values for global ranking
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print("\nGlobal Feature Importance via SHAP:")
    print(importance_df.to_string(index=False))

    # 5. Plot SHAP summary & Feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_sample, feature_names=FEATURE_COLUMNS, show=False)
    summary_plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot: {summary_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#1f77b4")
    plt.xlabel("Mean |SHAP Value| (Impact on Node Placement Score)")
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    importance_plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(importance_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot: {importance_plot_path}")

    # 6. Save Artifacts
    joblib.dump(model_score, os.path.join(output_dir, "placement_model.joblib"))
    joblib.dump(model_energy, os.path.join(output_dir, "energy_model.joblib"))
    joblib.dump(model_contention, os.path.join(output_dir, "contention_model.joblib"))
    joblib.dump(explainer, os.path.join(output_dir, "explainer.joblib"))

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "base_value": float(explainer.expected_value),
        "metrics": {
            "placement_score": {"r2": float(r2_s), "mae": float(mae_s), "rmse": float(rmse_s)},
            "energy_model": {"r2": float(r2_e)},
            "contention_model": {"r2": float(r2_c)},
        },
        "feature_importance": importance_df.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model artifacts and metadata saved to {output_dir}/")

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SHAP-CC scheduling models")
    parser.add_argument("--dataset", type=str, default="data/workload_features_dataset.csv")
    parser.add_argument("--output", type=str, default="model")
    args = parser.parse_args()
    train_and_evaluate(dataset_path=args.dataset, output_dir=args.output)
