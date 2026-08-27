import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

st.set_page_config(page_title="Loan Performance Intelligence", layout="wide")


@st.cache_data
def load_data():
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    static = pd.read_csv(config.STATIC_FILE)
    macro = pd.read_csv(config.MACRO_SCENARIOS_FILE)

    with open(config.VALIDATION_RULES_FILE) as f:
        rules = json.load(f)

    return train, test, static, macro, rules


def compute_psi(expected, actual, bins=10):
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    expected_counts = np.histogram(expected.dropna(), bins=breakpoints)[0]
    actual_counts = np.histogram(actual.dropna(), bins=breakpoints)[0]

    expected_pct = (expected_counts + 1) / (expected_counts.sum() + bins)
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def render_data_quality_tab(train, test):
    st.header("Data Quality Overview")

    col1, col2, col3 = st.columns(3)

    total_missing = train.isna().mean().mean() * 100
    quality_score = max(0, 100 - total_missing * 0.5)

    col1.metric("Train Rows", f"{len(train):,}")
    col2.metric("Test Rows", f"{len(test):,}")
    col3.metric("Quality Score", f"{quality_score:.1f} / 100")

    st.subheader("Missingness by Column")
    miss_data = pd.DataFrame({
        "Train Missing %": train.isna().mean() * 100,
        "Test Missing %": test.isna().mean() * 100
    }).sort_values("Train Missing %", ascending=False)

    miss_data = miss_data[miss_data["Train Missing %"] > 0]

    if not miss_data.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, len(miss_data) * 0.3)))
        y_pos = range(len(miss_data))
        ax.barh(y_pos, miss_data["Train Missing %"], height=0.4, label="Train", alpha=0.8, color="#2196F3")
        ax.barh([y + 0.4 for y in y_pos], miss_data["Test Missing %"], height=0.4, label="Test", alpha=0.8, color="#FF5722")
        ax.set_yticks([y + 0.2 for y in y_pos])
        ax.set_yticklabels(miss_data.index, fontsize=8)
        ax.set_xlabel("Missing %")
        ax.legend()
        ax.set_title("Missingness Comparison")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Numeric Distributions")
    num_cols = [c for c in config.NUMERIC_RAW if c in train.columns]
    if num_cols:
        stats = train[num_cols].describe().T[["mean", "std", "min", "max"]]
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)


def render_drift_tab(train, test):
    st.header("Feature Drift (PSI)")

    num_cols = [c for c in config.NUMERIC_RAW if c in train.columns and c in test.columns]

    if not num_cols:
        st.warning("No shared numeric columns found.")
        return

    psi_results = []
    for col in num_cols:
        tr_clean = train[col].dropna()
        te_clean = test[col].dropna()
        if len(tr_clean) > 0 and len(te_clean) > 0:
            psi_val = compute_psi(tr_clean, te_clean)
            mean_shift = abs(tr_clean.mean() - te_clean.mean())
            psi_results.append({
                "Feature": col,
                "PSI": psi_val,
                "Mean Shift": mean_shift,
                "Train Mean": tr_clean.mean(),
                "Test Mean": te_clean.mean(),
                "Status": "Stable" if psi_val < 0.1 else ("Moderate" if psi_val < 0.25 else "Significant")
            })

    psi_df = pd.DataFrame(psi_results).sort_values("PSI", ascending=False)
    st.dataframe(psi_df.style.format({"PSI": "{:.4f}", "Mean Shift": "{:.4f}", "Train Mean": "{:.4f}", "Test Mean": "{:.4f}"}),
                 use_container_width=True)

    st.subheader("Distribution Overlays")
    selected = st.selectbox("Select feature to compare", num_cols)

    if selected:
        fig, ax = plt.subplots(figsize=(10, 5))
        train[selected].dropna().hist(bins=40, alpha=0.6, label="Train", ax=ax, color="#2196F3", density=True)
        test[selected].dropna().hist(bins=40, alpha=0.6, label="Test", ax=ax, color="#FF5722", density=True)
        ax.set_title(f"Distribution: {selected}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_model_performance_tab(train):
    st.header("Model Performance")

    model_files = list(config.MODEL_DIR.glob("*_xgb_calibrated.joblib"))
    if not model_files:
        st.warning("No trained models found.")
        return

    metrics_data = []
    for mf in model_files:
        target_name = mf.stem.replace("_xgb_calibrated", "")
        metrics_data.append({
            "Target": target_name,
            "Model File": mf.name,
            "Size (KB)": mf.stat().st_size / 1024
        })

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    col1, col2 = st.columns(2)

    calib_path = config.REPORT_DIR / "calibration_curves.png"
    if calib_path.exists():
        col1.subheader("Calibration Curves")
        col1.image(str(calib_path))

    shap_path = config.REPORT_DIR / "shap_global_importance.png"
    if shap_path.exists():
        col2.subheader("SHAP Feature Importance")
        col2.image(str(shap_path))

    calib_seg_path = config.REPORT_DIR / "calibration_by_segment.png"
    if calib_seg_path.exists():
        st.subheader("Calibration by Segment")
        st.image(str(calib_seg_path))


def render_anomaly_tab(train, test):
    st.header("Anomaly Explorer")

    sub_path = config.SUBMISSION_OUTPUT
    if not sub_path.exists():
        st.warning("No submission.csv found. Run the pipeline first.")
        return

    st.info("Showing top anomalies detected in the test dataset (from submission.csv).")

    try:
        sub_df = pd.read_csv(sub_path)
        if "anomaly_score" not in sub_df.columns:
            st.error("anomaly_score column missing from submission.")
            return
            
        # Join with test to get raw features for display
        merged = pd.merge(sub_df, test, on=[config.COL_LOAN_ID], how="left")
        
        top_anomalies = merged.nlargest(20, "anomaly_score")

        display_cols = [config.COL_LOAN_ID, "anomaly_score", "predicted_exception_type", "recommended_action"]
        display_cols += [c for c in [config.COL_STATUS, config.COL_DPD, config.COL_CURR_BALANCE,
                                      config.COL_CREDIT_BAND, config.COL_INTEREST_RATE] if c in top_anomalies.columns]

        st.dataframe(top_anomalies[display_cols].reset_index(drop=True), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(sub_df["anomaly_score"].dropna(), bins=50, color="#673AB7", alpha=0.8, edgecolor="white")
        ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Test Set Anomaly Score Distribution")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Failed to display anomaly scores: {e}")


def render_scenario_tab():
    st.header("Scenario & Stress Simulation")

    scenario_path = config.REPORT_DIR / "scenario_report.md"
    if scenario_path.exists():
        st.markdown(scenario_path.read_text(encoding="utf-8"))
    else:
        st.warning("No scenario report found. Run the pipeline first.")

    mc_img = config.REPORT_DIR / "monte_carlo_loss_distribution.png"
    if mc_img.exists():
        st.subheader("Monte Carlo Loss Distribution")
        st.image(str(mc_img))

    mc_report = config.REPORT_DIR / "monte_carlo_report.md"
    if mc_report.exists():
        st.subheader("Monte Carlo Summary")
        st.markdown(mc_report.read_text(encoding="utf-8"))

    stress_report = config.REPORT_DIR / "stress_sensitivity_clusters.md"
    if stress_report.exists():
        st.subheader("Stress Sensitivity by Feature Cluster")
        st.markdown(stress_report.read_text(encoding="utf-8"))


def render_survival_tab():
    st.header("Survival Analysis")

    col1, col2 = st.columns(2)

    km_credit = config.REPORT_DIR / "km_curves_credit.png"
    if km_credit.exists():
        col1.subheader("KM Curves by Credit Band")
        col1.image(str(km_credit))

    km_ltv = config.REPORT_DIR / "km_curves_ltv.png"
    if km_ltv.exists():
        col2.subheader("KM Curves by LTV Band")
        col2.image(str(km_ltv))

    cif_path = config.REPORT_DIR / "competing_risk_cif.png"
    if cif_path.exists():
        st.subheader("Competing Risk: Default vs Prepayment CIF")
        st.image(str(cif_path))

    explainability_path = config.REPORT_DIR / "explainability_report.md"
    if explainability_path.exists():
        with st.expander("Full Explainability Report"):
            st.markdown(explainability_path.read_text(encoding="utf-8"))


def main():
    st.title("Loan Performance Intelligence Dashboard")
    st.caption("Real-time drift monitoring, model performance, anomaly detection, and scenario analysis")

    try:
        train, test, static, macro, rules = load_data()
    except Exception as e:
        st.error(f"Could not load data. Run the pipeline first: {e}")
        return

    st.sidebar.header("Pipeline Status")
    st.sidebar.metric("Train Dataset", f"{len(train):,} rows")
    st.sidebar.metric("Test Dataset", f"{len(test):,} rows")
    st.sidebar.metric("Models Trained", len(list(config.MODEL_DIR.glob("*.joblib"))))
    st.sidebar.metric("Reports Generated", len(list(config.REPORT_DIR.glob("*.md"))))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Available Reports**")

    reports = list(config.REPORT_DIR.glob("*.md"))
    if reports:
        for r in reports:
            st.sidebar.markdown(f"- {r.stem}")
    else:
        st.sidebar.markdown("*No reports generated yet.*")

    tabs = st.tabs([
        "Data Quality",
        "Feature Drift",
        "Model Performance",
        "Anomaly Explorer",
        "Scenarios & Stress",
        "Survival Analysis"
    ])

    with tabs[0]:
        render_data_quality_tab(train, test)

    with tabs[1]:
        render_drift_tab(train, test)

    with tabs[2]:
        render_model_performance_tab(train)

    with tabs[3]:
        render_anomaly_tab(train, test)

    with tabs[4]:
        render_scenario_tab()

    with tabs[5]:
        render_survival_tab()

    st.sidebar.header("Pipeline Status")
    st.sidebar.metric("Train Dataset", f"{len(train):,} rows")
    st.sidebar.metric("Test Dataset", f"{len(test):,} rows")
    st.sidebar.metric("Models Trained", len(list(config.MODEL_DIR.glob("*.joblib"))))
    st.sidebar.metric("Reports Generated", len(list(config.REPORT_DIR.glob("*.md"))))

    reports = list(config.REPORT_DIR.glob("*.md"))
    if reports:
        st.sidebar.subheader("Available Reports")
        for r in reports:
            st.sidebar.markdown(f"- {r.stem}")

    experiment_log = config.REPORT_DIR / "experiment_log.json"
    if experiment_log.exists():
        st.sidebar.subheader("Experiment History")
        with open(experiment_log) as f:
            exp_data = json.load(f)
        if "experiments" in exp_data:
            st.sidebar.metric("Total Experiments", len(exp_data["experiments"]))

    feedback_file = config.HUMAN_FEEDBACK_FILE
    if feedback_file.exists():
        st.sidebar.subheader("Active Learning")
        with open(feedback_file) as f:
            feedback = json.load(f)
        st.sidebar.metric("Feedback Samples", len(feedback))


if __name__ == "__main__":
    main()
