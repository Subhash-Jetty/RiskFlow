# Demo Script — 5‑Minute Walkthrough

> Matches the 15‑step flow from Section 14 of the Problem Statement.

---

## Step 1 — Dataset and Targets
"We are working with a monthly loan performance panel — one row per loan per month — merged with static origination attributes and servicer update records. Our target variables cover the full risk spectrum: 3‑month and 6‑month delinquency flags, a 12‑month default flag, a 12‑month prepayment flag, next‑state transition labels, and exception‑required / exception‑type labels for anomaly governance."

## Step 2 — Data Profiling Report
"Before touching any models I ran a full data intelligence pass. The `data_quality_report.md` contains column distributions, missingness heatmaps, Pearson and Cramér's V correlations, IQR/Z‑score outlier counts, and cross‑column relationship checks — for example verifying that `loan_age_months` is consistent with `origination_month`."

## Step 3 — Top Data‑Quality Issues
"The profiling surfaced three main issues: first, approximately 5 % of `document_status` values are missing; second, a handful of records violate the balance‑consistency rule where `current_balance` exceeds 105 % of `original_balance`; third, PSI drift analysis between training months and the held‑out test window flagged `interest_rate` and `days_past_due` as the columns with highest distributional shift."

## Step 4 — Feature‑Engineering Approach
"I engineered rolling DPD averages at 3‑, 6‑, and 12‑month windows using strictly historical expanding windows — never peeking forward. I also derived `balance_ratio`, `rate_spread` vs the portfolio median, `loan_age_pct`, a `servicer_conflict_flag` from cross‑system reconciliation, and smoothed target encodings of categorical bands. All fill‑values were computed on the training split only."

## Step 5 — Time‑Aware Split
"The split is purely chronological by `reporting_month`. The earliest 70 % of months form the training set, the next 15 % the validation set, and the final 15 % the test set. A single loan can appear in multiple splits — that is correct and expected for panel data. What I strictly prevented is any row‑level shuffle or look‑ahead feature computation."

## Step 6 — Baseline Model Performance
"Logistic Regression serves as the interpretable baseline. On the test split it achieves an ROC‑AUC and PR‑AUC that we log via MLflow alongside F1, recall‑at‑precision‑0.9, and Brier score. These numbers anchor the improvement story."

## Step 7 — Improved Model Performance
"XGBoost with early stopping and scale_pos_weight for class imbalance improves on every metric. We also apply Platt scaling calibration. All metrics — including macro‑F1 for the multiclass `next_state` and `exception_type` targets — are logged to MLflow and printed in the explainability report."

## Step 8 — Survival or Transition Model Output
"I fitted a Cox Proportional Hazards model using `lifelines`. Active loans are right‑censored; defaulted loans define the event. The concordance index is reported alongside a simple logistic‑regression baseline at a fixed 12‑month horizon. Kaplan‑Meier curves segmented by credit‑score band and LTV band visualize how survival diverges across risk tiers."

## Step 9 — Anomaly Examples
"An Isolation Forest produces a continuous anomaly score normalised to 0–1, blended with a rule‑violation score derived from `validation_rules.json`. The top 20 anomalies are presented reviewer‑ready, each annotated with its top 3 SHAP‑derived drivers — for instance, an unusually high `interest_rate` combined with a `document_status` of Missing and a rising `days_past_due` trend."

## Step 10 — Scenario Output
"Three macro scenarios — base, adverse‑credit, and high‑prepayment — are loaded from `macro_scenarios.csv`. Feature shifts are applied (rate bumps, credit tightening, HPI shocks) and the portfolio is re‑scored. The scenario report shows projected delinquency, default, and prepayment rates at the portfolio level and segmented by vintage year, credit band, state, and servicer."

## Step 11 — Local Explanation for One Loan
"I pick a single high‑risk loan from the test set and display its SHAP waterfall. The plot shows exactly how `rolling_dpd_mean_6m`, `balance_ratio`, and `credit_score_band` each pushed the default prediction above the baseline expected value."

## Step 12 — Automated Reviewer Note
"The copilot generates a structured note labelled RECOMMENDATION. It retrieves relevant field definitions from the FAISS‑indexed data dictionary, injects the SHAP drivers, and composes a two‑paragraph summary. If no API key is present the pipeline degrades to a deterministic template that still combines SHAP + RAG context."

## Step 13 — Example of LLM Output Rejected or Corrected
"I surface two concrete cases. In the first, the template note was overconfident — it flagged a loan as high‑risk despite moderate feature values; the human override was to downgrade the assessment. In the second, the note was vague — it omitted the second‑largest SHAP driver; the human correction was to explicitly insert that driver and its contribution."

## Step 14 — Final Submission File
"`submission.csv` matches the template column order exactly: `loan_id`, `month_index`, probability columns for delinquency/default/prepayment, predicted next state, exception probability, predicted exception type, anomaly score, top drivers, recommended action, and confidence score."

## Step 15 — AI Development Log
"The `ai_development_log.md` documents every tool used, representative prompts, accepted and rejected outputs, the human review process, the approximate AI‑generated code share (65 % boilerplate, 35 % human architecture/validation), and lessons learned around leakage prevention and graceful LLM fallbacks."
