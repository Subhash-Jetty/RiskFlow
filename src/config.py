"""
config.py – Single source of truth for column names, file paths, random seeds,
target definitions, feature lists, and hyper-parameter defaults.

Every other module imports from here so that names can never drift.
"""

from pathlib import Path
import os

# ──────────────────────────────────────────────
# 0. Project root & directory layout
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR, NOTEBOOK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Random seeds
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# 2. Data file paths
# ──────────────────────────────────────────────
TRAIN_FILE = DATA_DIR / "loan_monthly_performance_train.csv"
TEST_FILE = DATA_DIR / "loan_monthly_performance_test.csv"
STATIC_FILE = DATA_DIR / "loan_static_attributes.csv"
SERVICER_FILE = DATA_DIR / "servicer_updates.csv"
VALIDATION_RULES_FILE = DATA_DIR / "validation_rules.json"
MACRO_SCENARIOS_FILE = DATA_DIR / "macro_scenarios.csv"
SUBMISSION_TEMPLATE_FILE = DATA_DIR / "submission_template.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "data_dictionary.md"

SUBMISSION_OUTPUT = PROJECT_ROOT / "submission.csv"

# ──────────────────────────────────────────────
# 3. Canonical column names – monthly panel
# ──────────────────────────────────────────────
COL_LOAN_ID = "loan_id"
COL_MONTH_INDEX = "month_index"
COL_REPORTING_MONTH = "reporting_month"
COL_ORIGINATION_MONTH = "origination_month"
COL_LOAN_AGE = "loan_age_months"
COL_REMAINING_TERM = "remaining_term_months"
COL_ORIG_BALANCE = "original_balance"
COL_CURR_BALANCE = "current_balance"
COL_INTEREST_RATE = "interest_rate"
COL_CREDIT_BAND = "credit_score_band"
COL_LTV_BAND = "ltv_band"
COL_DTI_BAND = "dti_band"
COL_STATE = "state"
COL_LOAN_PURPOSE = "loan_purpose"
COL_OCCUPANCY = "occupancy_type"
COL_PROPERTY_TYPE = "property_type"
COL_SERVICER = "servicer_name"
COL_STATUS = "current_status"
COL_DPD = "days_past_due"
COL_MOD_FLAG = "modification_flag"
COL_PREPAY_FLAG = "prepayment_flag"
COL_DEFAULT_FLAG = "default_flag"
COL_LOSS_BAND = "loss_severity_band"
COL_LAST_UPDATED = "last_updated_at"
COL_SOURCE_SYSTEM = "source_system"
COL_DOC_STATUS = "document_status"

# ──────────────────────────────────────────────
# 4. Target columns
# ──────────────────────────────────────────────
TARGET_NEXT_3M_DEL = "next_3m_delinquency_flag"
TARGET_NEXT_6M_DEL = "next_6m_delinquency_flag"
TARGET_NEXT_12M_DEF = "next_12m_default_flag"
TARGET_NEXT_12M_PREPAY = "next_12m_prepayment_flag"
TARGET_NEXT_STATE = "next_state"
TARGET_EXCEPTION_REQ = "exception_required"
TARGET_EXCEPTION_TYPE = "exception_type"

BINARY_TARGETS = [
    TARGET_NEXT_3M_DEL,
    TARGET_NEXT_6M_DEL,
    TARGET_NEXT_12M_DEF,
    TARGET_NEXT_12M_PREPAY,
    TARGET_EXCEPTION_REQ,
]

MULTICLASS_TARGETS = [
    TARGET_NEXT_STATE,
    TARGET_EXCEPTION_TYPE,
]

ALL_TARGETS = BINARY_TARGETS + MULTICLASS_TARGETS

# ──────────────────────────────────────────────
# 5. Feature groups
# ──────────────────────────────────────────────
NUMERIC_RAW = [
    COL_LOAN_AGE, COL_REMAINING_TERM, COL_ORIG_BALANCE,
    COL_CURR_BALANCE, COL_INTEREST_RATE, COL_DPD,
    COL_MONTH_INDEX,
]

CATEGORICAL_RAW = [
    COL_CREDIT_BAND, COL_LTV_BAND, COL_DTI_BAND,
    COL_STATE, COL_LOAN_PURPOSE, COL_OCCUPANCY,
    COL_PROPERTY_TYPE, COL_SERVICER, COL_STATUS,
    COL_LOSS_BAND, COL_DOC_STATUS,
]

BINARY_RAW = [COL_MOD_FLAG, COL_PREPAY_FLAG, COL_DEFAULT_FLAG]

# Derived / engineered feature names
FEAT_BALANCE_RATIO = "balance_ratio"
FEAT_RATE_SPREAD = "rate_spread"           # vs median
FEAT_LOAN_AGE_PCT = "loan_age_pct"         # age / (age + remaining)
FEAT_ROLLING_DPD_3M = "rolling_dpd_mean_3m"
FEAT_ROLLING_DPD_6M = "rolling_dpd_mean_6m"
FEAT_ROLLING_DPD_12M = "rolling_dpd_mean_12m"
FEAT_ROLLING_DPD_STD_6M = "rolling_dpd_std_6m"
FEAT_DELINQ_COUNT = "delinquency_event_count"
FEAT_CONFLICT_FLAG = "servicer_conflict_flag"
FEAT_BALANCE_CHANGE = "balance_change_pct"
FEAT_MONTHS_SINCE_MOD = "months_since_modification"

DERIVED_FEATURES = [
    FEAT_BALANCE_RATIO, FEAT_RATE_SPREAD, FEAT_LOAN_AGE_PCT,
    FEAT_ROLLING_DPD_3M, FEAT_ROLLING_DPD_6M, FEAT_ROLLING_DPD_12M,
    FEAT_ROLLING_DPD_STD_6M, FEAT_DELINQ_COUNT, FEAT_CONFLICT_FLAG,
    FEAT_BALANCE_CHANGE, FEAT_MONTHS_SINCE_MOD,
]

# ──────────────────────────────────────────────
# 6. Categorical value domains (for synthetic data & encoding)
# ──────────────────────────────────────────────
CREDIT_BANDS = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
LTV_BANDS = ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]
DTI_BANDS = ["<=20", "21-30", "31-40", "41-50", ">50"]
STATES = [
    "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
LOAN_PURPOSES = ["Purchase", "Refinance", "Cash-Out Refinance"]
OCCUPANCY_TYPES = ["Primary", "Second Home", "Investment"]
PROPERTY_TYPES = ["Single Family", "Condo", "2-4 Unit", "Manufactured"]
SERVICER_NAMES = ["Servicer_A", "Servicer_B", "Servicer_C", "Servicer_D"]
STATUS_VALUES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid"]
NEXT_STATE_VALUES = STATUS_VALUES  # same domain
LOSS_BANDS = ["None", "0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
DOC_STATUSES = ["Complete", "Partial", "Missing", "Under Review"]
EXCEPTION_TYPES = ["None", "Balance_Discrepancy", "Status_Conflict",
                   "Document_Gap", "Payment_Anomaly"]
SOURCE_SYSTEMS = ["SystemA", "SystemB"]

# ──────────────────────────────────────────────
# 7. Hyper-parameter defaults
# ──────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=30,
)

XGB_MULTI_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
)

LOGREG_PARAMS = dict(
    max_iter=1000,
    random_state=RANDOM_SEED,
    solver="saga",
    class_weight="balanced",
    n_jobs=-1,
)

ISOLATION_FOREST_PARAMS = dict(
    n_estimators=200,
    contamination=0.05,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# ──────────────────────────────────────────────
# 8. Split ratios & SHAP sample size
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SHAP_SAMPLE_SIZE = 10_000

# ──────────────────────────────────────────────
# 9. Anomaly score blending
# ──────────────────────────────────────────────
ANOMALY_IF_WEIGHT = 0.7
ANOMALY_RULE_WEIGHT = 0.3

# ──────────────────────────────────────────────
# 10. Synthetic data parameters
# ──────────────────────────────────────────────
SYNTH_NUM_LOANS = 500
SYNTH_NUM_MONTHS = 100  # max months of history

# ──────────────────────────────────────────────
# 11. MLflow
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "intain_loan_performance"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# ──────────────────────────────────────────────
# 12. LLM / RAG settings
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAG_CHUNK_SIZE = 300       # characters per chunk
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# 13. Report paths
# ──────────────────────────────────────────────
DATA_QUALITY_REPORT = REPORT_DIR / "data_quality_report.md"
EXPLAINABILITY_REPORT = REPORT_DIR / "explainability_report.md"
SCENARIO_REPORT = REPORT_DIR / "scenario_report.md"
DEMO_SCRIPT = REPORT_DIR / "demo_script.md"
LLM_LOG_FILE = REPORT_DIR / "llm_logs.json"

# ──────────────────────────────────────────────
# 14. Submission template columns (expected order)
# ──────────────────────────────────────────────
SUBMISSION_COLUMNS = [
    "loan_id", "month_index",
    "next_3m_delinquency_prob", "next_6m_delinquency_prob",
    "next_12m_default_prob", "next_12m_prepayment_prob",
    "predicted_next_state", "exception_probability",
    "predicted_exception_type", "anomaly_score",
    "top_drivers", "recommended_action", "confidence_score",
]

# ──────────────────────────────────────────────
# 15. Monte Carlo simulation
# ──────────────────────────────────────────────
MONTE_CARLO_N_SIMULATIONS = 1000
MONTE_CARLO_VAR_PERCENTILES = [95, 99]

# ──────────────────────────────────────────────
# 16. Experiment runner
# ──────────────────────────────────────────────
EXPERIMENT_GRID = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.7),
    dict(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8),
    dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8),
    dict(n_estimators=700, max_depth=7, learning_rate=0.03, subsample=0.9),
    dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.7),
    dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.6),
]
EXPERIMENT_LOG_FILE = REPORT_DIR / "experiment_log.json"

# ──────────────────────────────────────────────
# 17. Feature store
# ──────────────────────────────────────────────
FEATURE_REGISTRY_FILE = MODEL_DIR / "feature_registry.json"

# ──────────────────────────────────────────────
# 18. Active learning
# ──────────────────────────────────────────────
ACTIVE_LEARNING_UNCERTAINTY_BAND = (0.35, 0.65)
ACTIVE_LEARNING_BATCH_SIZE = 50
HUMAN_FEEDBACK_FILE = DATA_DIR / "human_feedback.json"

# ──────────────────────────────────────────────
# 19. Dashboard
# ──────────────────────────────────────────────
DASHBOARD_PORT = 8501

# ──────────────────────────────────────────────
# 20. Stress sensitivity clustering
# ──────────────────────────────────────────────
CLUSTER_CORR_THRESHOLD = 0.5
