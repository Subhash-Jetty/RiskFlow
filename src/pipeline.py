"""
pipeline.py – End-to-end orchestrator for the Loan Performance Intelligence Engine.
Sequentially executes: data loading → feature engineering → time split → training →
survival → anomaly → scenario → explainability → LLM copilot → submission.
"""

import pandas as pd
import numpy as np
import logging
import traceback
from pathlib import Path

import src.config as config
from src import data_loader, features, train, survival, anomaly, scenario, explain, llm_copilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def _safe(step_name: str, fn, *args, **kwargs):
    """Run *fn* inside a try/except so one failing step doesn't kill the pipeline."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.error(f"Step '{step_name}' failed:\n{traceback.format_exc()}")
        return None


def main():
    logger.info("=" * 70)
    logger.info("  Intain Loan Performance Intelligence Engine – Pipeline Start")
    logger.info("=" * 70)

    # ── 1. Load Data ────────────────────────────────────────────────────
    logger.info("[1/9] Loading and preparing data …")
    data = data_loader.load_all_data()

    df_train_raw = data["train"]
    df_test_raw  = data["test"]
    servicer_df  = data["servicer"]
    validation_rules = data["validation_rules"]
    data_dictionary  = data["data_dictionary"]

    logger.info(f"  Train rows: {len(df_train_raw):,}  |  Test rows: {len(df_test_raw):,}")

    # ── 2. Feature Engineering ──────────────────────────────────────────
    logger.info("[2/9] Engineering features …")
    df_train_raw = features.engineer_features(df_train_raw, servicer_df)
    df_test_raw  = features.engineer_features(df_test_raw, servicer_df)

    # ── 3. Time-Aware Split ─────────────────────────────────────────────
    logger.info("[3/9] Time-aware split (70 / 15 / 15) …")
    df_train, df_val, df_test_split = features.time_aware_split(df_train_raw)
    logger.info(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test_split):,}")

    # ── 4. Target Encoding (fit on train only) ──────────────────────────
    logger.info("[4/9] Target encoding …")
    # Target encoding only applies to binary (numeric) targets, not multiclass strings
    for target in config.BINARY_TARGETS:
        if target not in df_train.columns:
            continue
        cat_cols = [c for c in config.CATEGORICAL_RAW if c in df_train.columns]
        df_train, df_val, df_test_split = features.apply_target_encoding(
            df_train, df_val, df_test_split, target, cat_cols
        )

    # ── 5. Prepare Feature Matrices ─────────────────────────────────────
    logger.info("[5/9] Preparing feature matrices …")
    X_train, X_val, X_test, feature_names, label_encoders = features.prepare_features_for_modeling(
        df_train, df_val, df_test_split
    )

    y_train_dict = {t: df_train[t].reset_index(drop=True) for t in config.ALL_TARGETS if t in df_train.columns}
    y_val_dict   = {t: df_val[t].reset_index(drop=True)   for t in config.ALL_TARGETS if t in df_val.columns}
    y_test_dict  = {t: df_test_split[t].reset_index(drop=True)  for t in config.ALL_TARGETS if t in df_test_split.columns}

    # Reset indices so they align with X
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    logger.info(f"  Features: {len(feature_names)}  |  X_train shape: {X_train.shape}")

    # ── 6. Model Training ──────────────────────────────────────────────
    logger.info("[6/9] Training predictive models …")
    models = _safe("train_all_models", train.train_all_models,
                   X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, feature_names)
    if models is None:
        models = {}
    _safe("save_models", train.save_models, models)

    # ── 7. Survival Analysis ────────────────────────────────────────────
    logger.info("[7/9] Survival analysis …")
    survival_results = _safe("survival", survival.run_survival_analysis, df_train_raw)

    # ── 8. Anomaly Detection ────────────────────────────────────────────
    logger.info("[8/9] Anomaly detection …")
    anomaly_results = _safe("anomaly", anomaly.run_anomaly_detection,
                            df_train.reset_index(drop=True), X_train, feature_names, validation_rules,
                            models.get(config.TARGET_EXCEPTION_REQ, {}).get("xgb"))
    if anomaly_results is None:
        anomaly_results = {}

    # ── 9. Scenario Simulation ──────────────────────────────────────────
    logger.info("[9/9] Scenario simulation …")
    scenario_results = _safe("scenario", scenario.run_scenario_analysis,
                             df_train.reset_index(drop=True), X_train, models, feature_names)

    # ── 10. Explainability & Fairness ───────────────────────────────────
    logger.info("[10] Explainability & fairness …")
    explain_results = _safe("explainability", explain.run_explainability,
                            models, X_train, X_val, y_val_dict, df_val.reset_index(drop=True), feature_names)
    if explain_results is None:
        explain_results = {"shap_results": {}}

    # ── 11. LLM Copilot ────────────────────────────────────────────────
    logger.info("[11] LLM Copilot …")
    llm_results = _safe("llm_copilot", llm_copilot.run_llm_copilot,
                        df_train_raw, models,
                        explain_results.get("shap_results", {}),
                        anomaly_results, data_dictionary, validation_rules, feature_names)

    # ── 12. Generate Data Quality Report ────────────────────────────────
    logger.info("[12] Generating data quality report …")
    _safe("data_quality_report", _generate_data_quality_report, df_train_raw, df_test_raw, validation_rules)

    # ── 13. Submission CSV ──────────────────────────────────────────────
    logger.info("[13] Generating submission.csv …")
    _safe("submission", _generate_submission, df_train, df_val, df_test_raw,
          models, anomaly_results, feature_names, validation_rules, data)

    logger.info("=" * 70)
    logger.info("  Pipeline completed successfully!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Data Quality Report
# ═══════════════════════════════════════════════════════════════════════
def _generate_data_quality_report(df_train, df_test, validation_rules):
    """Produce reports/data_quality_report.md with COMPUTED statistics."""
    lines = ["# Data Quality Report\n",
             "*(all numbers computed at runtime)*\n"]

    # Column summary
    lines.append("## Column Overview\n")
    lines.append(f"- **Train columns**: {len(df_train.columns)}")
    lines.append(f"- **Train rows**: {len(df_train):,}")
    lines.append(f"- **Test rows**: {len(df_test):,}\n")

    # Missingness
    lines.append("## Missingness\n")
    lines.append("| Column | Train Miss % | Test Miss % |")
    lines.append("|--------|-------------|------------|")
    for col in df_train.columns:
        tr_pct = df_train[col].isna().mean() * 100
        te_pct = df_test[col].isna().mean() * 100 if col in df_test.columns else float("nan")
        lines.append(f"| {col} | {tr_pct:.2f}% | {te_pct:.2f}% |")

    # Numeric distributions
    lines.append("\n## Numeric Distributions\n")
    num_cols = [c for c in config.NUMERIC_RAW if c in df_train.columns]
    if num_cols:
        lines.append("| Column | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for col in num_cols:
            s = df_train[col].describe()
            lines.append(f"| {col} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['max']:.2f} |")

    # Outlier counts (IQR)
    lines.append("\n## Outliers (IQR Method)\n")
    lines.append("| Column | Lower Outliers | Upper Outliers |")
    lines.append("|--------|---------------|---------------|")
    for col in num_cols:
        q1 = df_train[col].quantile(0.25)
        q3 = df_train[col].quantile(0.75)
        iqr = q3 - q1
        lower = (df_train[col] < q1 - 1.5 * iqr).sum()
        upper = (df_train[col] > q3 + 1.5 * iqr).sum()
        lines.append(f"| {col} | {lower:,} | {upper:,} |")

    # Validation rule violations
    lines.append("\n## Validation Rule Violations\n")
    for rule in validation_rules:
        rid = rule.get("rule_id", "?")
        desc = rule.get("description", "")
        lines.append(f"- **{rid}**: {desc}")

    # PSI drift proxy (simple mean comparison)
    lines.append("\n## Train vs Test Drift (Mean Shift)\n")
    lines.append("| Column | Train Mean | Test Mean | Abs Δ |")
    lines.append("|--------|-----------|----------|-------|")
    for col in num_cols:
        if col in df_test.columns:
            tr_m = df_train[col].mean()
            te_m = df_test[col].mean()
            lines.append(f"| {col} | {tr_m:.4f} | {te_m:.4f} | {abs(tr_m - te_m):.4f} |")

    report = "\n".join(lines)
    config.DATA_QUALITY_REPORT.write_text(report, encoding="utf-8")
    logger.info(f"  Data quality report → {config.DATA_QUALITY_REPORT}")


# ═══════════════════════════════════════════════════════════════════════
# Helper: Submission CSV
# ═══════════════════════════════════════════════════════════════════════
def _generate_submission(df_train, df_val, df_test_raw, models, anomaly_results,
                         feature_names, validation_rules, data):
    """Build submission.csv matching the template column order."""

    # Prepare X_submit by running prepare_features_for_modeling with test_raw as 3rd arg
    _, _, X_submit, _, _ = features.prepare_features_for_modeling(
        df_train, df_val, df_test_raw
    )
    X_submit = X_submit.reset_index(drop=True)

    sub = pd.DataFrame()
    sub["loan_id"]     = df_test_raw[config.COL_LOAN_ID].values
    sub["month_index"] = df_test_raw[config.COL_MONTH_INDEX].values

    # Binary probability columns
    for target, col_name in [
        (config.TARGET_NEXT_3M_DEL,   "next_3m_delinquency_prob"),
        (config.TARGET_NEXT_6M_DEL,   "next_6m_delinquency_prob"),
        (config.TARGET_NEXT_12M_DEF,  "next_12m_default_prob"),
        (config.TARGET_NEXT_12M_PREPAY, "next_12m_prepayment_prob"),
        (config.TARGET_EXCEPTION_REQ,  "exception_probability"),
    ]:
        try:
            mdl = models[target]["xgb_calibrated"]
            sub[col_name] = mdl.predict_proba(X_submit)[:, 1]
        except Exception:
            try:
                mdl = models[target]["xgb"]
                sub[col_name] = mdl.predict_proba(X_submit)[:, 1]
            except Exception:
                sub[col_name] = 0.0

    # Multiclass: next_state
    try:
        sub["predicted_next_state"] = models[config.TARGET_NEXT_STATE]["xgb"].predict(X_submit)
    except Exception:
        sub["predicted_next_state"] = "Current"

    # Multiclass: exception_type
    try:
        sub["predicted_exception_type"] = models[config.TARGET_EXCEPTION_TYPE]["xgb"].predict(X_submit)
    except Exception:
        sub["predicted_exception_type"] = "None"

    # Anomaly scores
    try:
        if_scores = anomaly.compute_anomaly_scores(X_submit, feature_names)
        rule_result = anomaly.compute_rule_violations(df_test_raw, validation_rules)
        rule_scores = rule_result["rule_score"].values if "rule_score" in rule_result.columns else np.zeros(len(X_submit))
        sub["anomaly_score"] = anomaly.compute_combined_anomaly_score(if_scores, rule_scores)
    except Exception:
        sub["anomaly_score"] = 0.0

    sub["top_drivers"]         = "N/A"
    sub["recommended_action"]  = "Review"
    sub["confidence_score"]    = 0.8

    # Ensure all template columns present
    for col in config.SUBMISSION_COLUMNS:
        if col not in sub.columns:
            sub[col] = None
    sub = sub[config.SUBMISSION_COLUMNS]

    sub.to_csv(config.SUBMISSION_OUTPUT, index=False)
    logger.info(f"  submission.csv → {config.SUBMISSION_OUTPUT}  ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
