# Development Log

## 1. Overview

This document captures the technical decisions, debugging steps, and iterative refinements I made while building the Loan Performance Intelligence Engine. It covers architecture choices, leakage prevention, model calibration, and the integration of the automated reviewer copilot.

---

## 2. Tools & Libraries Used

| Tool | Purpose | Notes |
|------|---------|-------|
| Python 3.14 | Core language | All modules in `src/` |
| XGBoost 3.4.1 | Gradient boosted trees for predictive modelling | Binary + multiclass targets |
| scikit-learn 1.9 | Logistic Regression, Isolation Forest, calibration | Used CalibratedClassifierCV |
| lifelines 0.30.3 | CoxPH survival model, Kaplan-Meier curves | Duration/event censoring |
| SHAP 0.52.0 | TreeExplainer for global/local feature importance | Waterfall + bar plots |
| FAISS | Vector similarity for data dictionary retrieval | IndexFlatIP with L2 normalisation |
| Sentence-Transformers | Embedding data-dictionary chunks for the FAISS RAG index | `all-MiniLM-L6-v2` |

---

## 3. Key Technical Decisions

### Decision 1 — Rolling DPD Feature Design

**Problem**: Needed a 6-month rolling mean of `days_past_due` per loan using only historical rows.

**Initial approach**: `.groupby('loan_id')['days_past_due'].transform(lambda x: x.rolling(6).mean())`.

**Issue**: `.rolling(6)` is a fixed trailing window over the sorted index, but it does not guarantee strictly historical-only computation if the dataframe isn't pre-sorted by time within each loan. More importantly, it includes the current row.

**Solution**: Rewrote to use `.expanding().mean().shift(1)` after sorting by `reporting_month` within each group, which guarantees: (a) only past data is used, and (b) the current row is excluded via `shift(1)`.

### Decision 2 — Target Encoding Leakage Prevention

**Problem**: Target encoding categorical features could leak validation/test target distributions into training features.

**Solution**: Structured the code so the encoding map is fitted exclusively on `df_train`, then the mapping is applied (transform-only) to `df_val` and `df_test`. Unseen categories default to the global mean. Smoothing factor set to 10.

### Decision 3 — Calibration Strategy

**Problem**: Needed well-calibrated probabilities for binary and multiclass targets.

**Solution**: Used Platt scaling (`method='sigmoid'`) for binary targets, and isotonic regression (`method='isotonic'`) for multiclass targets. Isotonic is preferred for multiclass because sigmoid assumes binary log-odds. Compared Brier scores on validation set to confirm improvement.

**Compatibility note**: `CalibratedClassifierCV(cv='prefit')` was removed in scikit-learn 1.9, so I added a try/except fallback to `cv=3`.

### Decision 4 — Isolation Forest Score Normalisation

**Problem**: Raw `decision_function()` output returns negative values for anomalies, which is counterintuitive for a "risk score."

**Solution**: Inverted the scale so that higher values = more anomalous. Final formula: `1 - (score - min) / (max - min)`.

### Decision 5 — Reviewer Note Architecture

**Problem**: Needed structured, grounded reviewer notes for flagged loans.

**Solution**: Built a template-first approach that injects only verified SHAP contributions and model scores, labelled as RECOMMENDATION. The system retrieves relevant field definitions from the FAISS-indexed data dictionary, injects the SHAP drivers, and composes a structured summary. If no API key is present, the pipeline degrades to a deterministic template that still combines SHAP + RAG context.

---

## 4. Review Process

My review workflow for every code module:

1. **Check imports** — Verify everything references `src.config` constants, not hardcoded strings.
2. **Leakage audit** — For any feature that touches target columns or future rows, manually trace the data flow to confirm no look-ahead.
3. **Run on synthetic data** — Execute the function in isolation on a 500-loan synthetic dataset to confirm shapes, dtypes, and no NaN explosions.
4. **Metric sanity check** — Verify that reported metrics (AUC, F1, Brier) come from the held-out validation split, not training data.

---

## 5. Bugs Encountered & Fixed

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | `CalibratedClassifierCV(cv='prefit')` crash | Removed in sklearn 1.9 | Added try/except fallback to `cv=3` |
| 2 | `LogisticRegression(multi_class=...)` crash | Removed in sklearn 1.9 | Removed parameter |
| 3 | Target encoding crash on multiclass string targets | TE applied to `next_state` which has string values | Limited TE to binary targets only |
| 4 | Survival `prepare_survival_data` called twice | Pipeline and module both called it | Fixed pipeline to call `run_survival_analysis(df)` directly |
| 5 | Anomaly shape mismatch (21980 vs 20118 rows) | Passed `df_train_raw` instead of `df_train` | Changed to pass split-aligned dataframe |
| 6 | KM curves NaN | Missing duration/event values | Added `dropna(subset=['duration','event'])` and `clip(lower=1)` |
| 7 | Rolling DPD features all identical | Used `.expanding()` for all windows | Fixed to use actual `.rolling(window=3/6/12)` |
| 8 | `pct_change()` FutureWarning | Deprecated `fill_method` param | Added `fill_method=None` |
| 9 | Scenario segment analysis returning N/A | Index mismatch between df and X | Passed `reset_index(drop=True)` aligned data |
| 10 | Explainability report crash | Missing `tabulate` dependency | Replaced `.to_markdown()` with `.to_string()` |

---

## 6. Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Rolling feature window | `.expanding().mean().shift(1)` | Prevents future leakage and current-row contamination |
| Target encoding scope | Fit on train only | Prevents val/test leakage |
| Multiclass calibration | Isotonic | Sigmoid assumes binary log-odds; isotonic is non-parametric |
| Reviewer output | Structured template + verified numbers | Prevents hallucinated statistics |
| Anomaly score direction | Inverted (higher = anomalous) | Matches business expectation for a "risk score" |
| Config architecture | Single `config.py` source of truth | Every column name, target, hyperparameter in one place |
| Error handling | `_safe()` wrapper in pipeline | Any step failure is caught and logged; pipeline continues |

---

## 7. Lessons Learned

1. **Config-driven architecture prevents drift.** Having a single `config.py` that defines every column name, target list, and hyperparameter meant no inconsistent names across modules.

2. **Graceful degradation is non-negotiable.** The reviewer copilot must work without an API key. Building the template fallback first and treating the LLM call as an optional upgrade saved hours of debugging.

3. **Synthetic data is harder than it looks.** Generating realistic loan status transitions (Current → 30DPD → 60DPD → Default, with plausible cure rates) required manual calibration of transition probabilities.

4. **Prompt logging builds trust.** Saving every prompt and output to `llm_logs.json` provides a transparent audit trail.

5. **Time-aware splits are critical for financial time-series.** Random splits would leak future information into training, inflating metrics artificially.
