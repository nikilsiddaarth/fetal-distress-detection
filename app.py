import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
 
# ============================================================
# FETOGUARD AI — DUAL-MODEL PREMIUM UI
# Explainable & Risk-Aware Fetal State Classification from CTG
#
# Uses ONLY the existing trained models + existing processed data.
# No retraining. No modification of models/dataset.
# ============================================================
 
st.set_page_config(
    page_title="FetoGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
BASE_DIR = Path(__file__).resolve().parent
RF_PATH = BASE_DIR / "models" / "random_forest.pkl"
XGB_PATH = BASE_DIR / "models" / "xgboost.pkl"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
YTEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"
MODEL_COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.csv"
RF_IMPORTANCE_PATH = BASE_DIR / "results" / "random_forest_feature_importance.csv"
XGB_IMPORTANCE_PATH = BASE_DIR / "results" / "xgboost_feature_importance.csv"
 
CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathological"}
CLASS_ICONS = {1: "●", 2: "▲", 3: "◆"}
CLASS_COLORS = {1: "#38d39f", 2: "#f7b84b", 3: "#ff5c69"}
STATE_ORDER = [1, 2, 3]  # Normal, Suspect, Pathological — fixed display order
 
# Columns present in the current 41-column feature set that are FIGO/SisPorto
# morphologic-PATTERN ANNOTATIONS, not raw physiological CTG signal features.
# They are kept exactly as-is (nothing dropped, per project constraints) but
# flagged for transparency — the project's own notebook/05_leakage_check.ipynb
# already identifies this same set as leakage-prone.
ANNOTATION_COLUMNS = {"CLASS", "SUSP", "LD", "FS", "A", "B", "C", "D", "E", "AD", "DE"}
METADATA_COLUMNS = {"b", "e", "DR"}  # recording-segment metadata, not signal features
 
# ============================================================
# PREMIUM CSS
# ============================================================
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 82% 3%, rgba(80, 95, 255, 0.13), transparent 28%),
        radial-gradient(circle at 5% 35%, rgba(0, 210, 190, 0.08), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(120, 90, 255, 0.06), transparent 30%),
        #080d18;
}
[data-testid="stHeader"] { background: rgba(8,13,24,0.82); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #0a0f1b 100%);
    border-right: 1px solid rgba(148,163,184,0.12);
}
[data-testid="stSidebar"] * { color: #e7edf7 !important; }
.block-container { max-width: 1420px; padding-top: 2.0rem; padding-bottom: 4rem; }
 
h1, h2, h3, h4, p, label, .stMarkdown { color: #edf3fb; }
h2 { letter-spacing: -0.02em; }
.muted { color: #93a4bd !important; }
 
/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    border: 1px solid rgba(130,150,255,0.24); border-radius: 28px;
    padding: 42px 46px; margin-bottom: 22px;
    background:
        radial-gradient(circle at 88% 25%, rgba(91,91,255,0.24), transparent 30%),
        radial-gradient(circle at 68% 110%, rgba(0,212,190,0.15), transparent 32%),
        linear-gradient(135deg, #111a32 0%, #0d1528 48%, #101936 100%);
    box-shadow: 0 25px 70px rgba(0,0,0,0.30);
}
.hero:after {
    content: ""; position: absolute; width: 300px; height: 300px;
    right: -110px; top: -130px; border-radius: 50%;
    border: 1px solid rgba(120,140,255,0.16);
    box-shadow: 0 0 0 25px rgba(120,140,255,0.035), 0 0 0 55px rgba(120,140,255,0.025);
}
.hero-status {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px; border-radius: 999px;
    background: rgba(56,211,159,0.10); border: 1px solid rgba(56,211,159,0.30);
    color: #7ce8c4 !important; font-size: 0.74rem; font-weight: 800;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 18px;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #38d39f; display: inline-block;
    box-shadow: 0 0 0 3px rgba(56,211,159,0.20); animation: pulse 2.2s infinite; }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.4;} 100%{opacity:1;} }
.hero-kicker { color: #72e6d7; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 12px; }
.hero-title { font-size: 3.2rem; line-height: 1.03; font-weight: 900; letter-spacing: -0.045em; color: #ffffff !important; margin: 0; }
.hero-title span { background: linear-gradient(90deg, #ffffff, #9da8ff, #72e6d7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { margin-top: 15px; max-width: 760px; color: #a9b8cf !important; font-size: 1.05rem; line-height: 1.7; }
.pipeline-wrap { display: flex; align-items: stretch; gap: 6px; margin-top: 28px; flex-wrap: wrap; }
.pipe-card { flex: 1; min-width: 130px; border: 1px solid rgba(148,163,184,0.15); background: rgba(255,255,255,0.035); border-radius: 16px; padding: 14px 13px; transition: all 0.25s ease; }
.pipe-card:hover { background: rgba(114,230,215,0.07); border-color: rgba(114,230,215,0.35); transform: translateY(-3px); }
.pipe-num { color: #72e6d7 !important; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em; }
.pipe-label { color: #f2f6fc !important; font-size: 0.86rem; font-weight: 750; margin-top: 5px; }
.pipe-arrow { display: flex; align-items: center; color: #4a5a78; font-size: 1.1rem; padding: 0 2px; }
 
/* ---------- Notice ---------- */
.notice { border-radius: 15px; padding: 13px 17px; margin: 16px 0 24px; color: #b9c7dc !important; background: rgba(255,255,255,0.035); border: 1px solid rgba(148,163,184,0.14); font-size: 0.84rem; line-height: 1.55; }
.notice-warn { border-radius: 15px; padding: 14px 18px; margin: 8px 0 24px; color: #ffd9a8 !important; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.30); font-size: 0.84rem; line-height: 1.6; }
 
/* ---------- Section heading ---------- */
.section-kicker { color: #72e6d7 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.16em; font-weight: 800; }
.section-title { color: #f7faff !important; font-size: 1.7rem; font-weight: 850; margin-top: 2px; margin-bottom: 5px; }
 
/* ---------- Cards ---------- */
.glass-card { border: 1px solid rgba(148,163,184,0.13); background: rgba(17,25,43,0.72); border-radius: 20px; padding: 22px; box-shadow: 0 15px 45px rgba(0,0,0,0.16); transition: all 0.2s ease; }
.glass-card:hover { border-color: rgba(130,150,255,0.30); transform: translateY(-2px); }
.why-num { color: #4a5a78 !important; font-size: 1.6rem; font-weight: 900; }
.why-title { color: #fff !important; font-size: 1.12rem; font-weight: 800; margin: 6px 0 8px; }
.why-desc { color: #93a4bd !important; font-size: 0.88rem; line-height: 1.6; }
 
.stat-card { min-height: 116px; border: 1px solid rgba(148,163,184,0.13); background: linear-gradient(145deg, rgba(23,33,55,0.90), rgba(13,20,35,0.88)); border-radius: 19px; padding: 19px 20px; }
.stat-label { color: #8fa1bb !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.10em; font-weight: 750; }
.stat-value { color: #f8fbff !important; font-size: 1.55rem; font-weight: 850; margin-top: 7px; }
.stat-note { color: #72849f !important; font-size: 0.76rem; margin-top: 3px; }
 
/* ---------- Single-model prediction cards ---------- */
.model-card {
    border-radius: 22px; padding: 22px 24px; height: 100%;
    background: linear-gradient(160deg, rgba(23,33,55,0.9), rgba(11,17,30,0.9));
    border: 1px solid rgba(148,163,184,0.16);
}
.model-card-rf { border-top: 3px solid #7779ff; }
.model-card-xgb { border-top: 3px solid #72e6d7; }
.model-card-head { display:flex; align-items:center; gap:10px; margin-bottom: 14px; }
.model-card-title { color: #fff !important; font-size: 1.02rem; font-weight: 850; }
.model-card-sub { color: #8fa1bb !important; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; }
.model-card-state { font-size: 1.55rem; font-weight: 900; margin: 6px 0 2px; }
.model-card-conf { color: #b9c7dc !important; font-size: 0.9rem; }
 
/* ---------- Agreement banner ---------- */
.agree-banner {
    text-align: center; border-radius: 16px; padding: 14px 18px; margin: 16px 0;
    font-weight: 800; font-size: 1.0rem;
}
.agree-yes { background: rgba(56,211,159,0.10); border: 1px solid rgba(56,211,159,0.35); color: #7ce8c4 !important; }
.agree-no { background: rgba(255,92,105,0.10); border: 1px solid rgba(255,92,105,0.35); color: #ffb3ba !important; }
.agree-sub { font-weight: 500; font-size: 0.85rem; color: #b9c7dc !important; margin-top: 3px; }
 
/* ---------- Dual model final card ---------- */
.dual-card {
    border-radius: 26px; padding: 30px 32px; margin: 6px 0 20px;
    background: linear-gradient(135deg, rgba(91,93,245,0.14), rgba(114,230,215,0.06));
    border: 1px solid rgba(130,150,255,0.35);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
}
.dual-kicker { color: #9da8ff !important; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
.dual-state { font-size: 2.1rem; font-weight: 900; color: #fff !important; margin: 8px 0 2px; }
.dual-conf { color: #cdd6e8 !important; font-size: 1.0rem; }
 
/* ---------- Result ---------- */
.result-normal, .result-suspect, .result-path { border-radius: 22px; padding: 22px 24px; margin: 10px 0 17px; }
.result-normal { background: linear-gradient(135deg, rgba(16,185,129,.13), rgba(16,185,129,.045)); border: 1px solid rgba(52,211,153,.35); }
.result-suspect { background: linear-gradient(135deg, rgba(245,158,11,.14), rgba(245,158,11,.045)); border: 1px solid rgba(251,191,36,.38); }
.result-path { background: linear-gradient(135deg, rgba(239,68,68,.15), rgba(239,68,68,.045)); border: 1px solid rgba(248,113,113,.38); }
 
/* ---------- Model Arena ---------- */
.arena-card { border-radius: 24px; padding: 28px; background: linear-gradient(160deg, rgba(23,33,55,0.9), rgba(11,17,30,0.9)); border: 1px solid rgba(148,163,184,0.14); position: relative; }
.arena-winner { border: 1px solid rgba(114,230,215,0.45); box-shadow: 0 0 0 1px rgba(114,230,215,0.10), 0 20px 60px rgba(0,0,0,0.35); }
.arena-badge { position: absolute; top: -12px; right: 22px; background: linear-gradient(135deg, #38d39f, #72e6d7); color: #06251c !important; font-weight: 900; font-size: 0.7rem; padding: 5px 12px; border-radius: 999px; letter-spacing: 0.06em; }
.arena-name { color: #fff !important; font-size: 1.3rem; font-weight: 900; margin-bottom: 14px; }
.arena-metric-row { display: flex; gap: 22px; margin-top: 6px; }
.arena-metric-val { font-size: 2rem; font-weight: 900; color: #f8fbff !important; }
.arena-metric-label { color: #8fa1bb !important; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; }
.vs-badge { display: flex; align-items: center; justify-content: center; font-weight: 900; color: #4a5a78 !important; font-size: 0.95rem; }
 
/* ---------- Feature guide ---------- */
.feat-row { border: 1px solid rgba(148,163,184,0.12); border-radius: 14px; padding: 13px 16px; margin-bottom: 9px; background: rgba(255,255,255,0.025); }
.feat-tag { display: inline-block; font-weight: 900; color: #72e6d7 !important; background: rgba(114,230,215,0.08); border: 1px solid rgba(114,230,215,0.25); border-radius: 8px; padding: 2px 9px; font-size: 0.82rem; margin-right: 10px; }
.feat-flag { display: inline-block; font-weight: 800; color: #ffb974 !important; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); border-radius: 8px; padding: 2px 9px; font-size: 0.7rem; margin-left: 8px; }
.feat-desc { color: #c3cee1 !important; font-size: 0.88rem; margin-top: 4px; }
 
/* ---------- What-if delta chip ---------- */
.delta-chip { display:inline-block; padding: 3px 11px; border-radius: 999px; font-weight: 800; font-size: 0.78rem; margin-left: 8px; }
.delta-up { background: rgba(255,92,105,0.12); color: #ffb3ba !important; border: 1px solid rgba(255,92,105,0.3); }
.delta-down { background: rgba(56,211,159,0.12); color: #7ce8c4 !important; border: 1px solid rgba(56,211,159,0.3); }
.delta-flat { background: rgba(148,163,184,0.1); color: #b9c7dc !important; border: 1px solid rgba(148,163,184,0.25); }
 
/* ---------- Sidebar branding ---------- */
.brand { padding: 8px 2px 18px; }
.brand-name { font-size: 1.35rem; font-weight: 900; color: #fff !important; }
.brand-sub { color: #71829b !important; font-size: 0.77rem; margin-top: 3px; }
.status-line { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #b8c5d8 !important; margin: 4px 0; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #38d39f; box-shadow: 0 0 0 3px rgba(56,211,159,0.18); }
.status-dot-warn { background: #f7b84b; box-shadow: 0 0 0 3px rgba(247,184,75,0.18); }
 
/* ---------- Buttons ---------- */
.stButton > button { border-radius: 13px; min-height: 48px; font-weight: 800; border: 1px solid rgba(130,145,255,.35); background: linear-gradient(135deg, #5b5df5, #7779ff); color: white; box-shadow: 0 8px 25px rgba(91,93,245,.22); transition: all 0.18s ease; }
.stButton > button:hover { border-color: #8f91ff; transform: translateY(-1px); box-shadow: 0 12px 32px rgba(91,93,245,.32); }
 
/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #11192b !important; border-color: #273650 !important; border-radius: 11px !important; }
input { color: #edf3fb !important; }
.stRadio label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label { color: #b8c5d8 !important; }
 
/* ---------- Footer ---------- */
.footer { text-align: center; color: #61738d !important; font-size: 0.78rem; padding: 30px 0 10px; }
 
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# DATA / MODEL LOADING (defensive — every optional file can be missing)
# ============================================================
 
@st.cache_resource
def load_models():
    if not RF_PATH.exists():
        raise FileNotFoundError(f"Missing Random Forest model file: {RF_PATH.name}")
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost model file: {XGB_PATH.name}")
    return joblib.load(RF_PATH), joblib.load(XGB_PATH)
 
@st.cache_data
def load_test_data():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing processed test data: {TEST_PATH.name}")
    return pd.read_csv(TEST_PATH)
 
@st.cache_data
def load_optional_csv(path):
    """Load an optional CSV. Returns None (never raises) if missing/unreadable."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None
 
load_error = None
rf_model = xgb_model = test_df = None
try:
    rf_model, xgb_model = load_models()
    test_df = load_test_data()
except Exception as e:
    load_error = str(e)
 
if load_error:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">System error</div>'
        '<div class="hero-title">FetoGuard <span>could not start</span></div></div>',
        unsafe_allow_html=True,
    )
    st.error("FetoGuard could not load one or more REQUIRED project files.")
    st.code(load_error)
    st.info(
        "These files are required and must exist relative to app.py:\n"
        f"  {RF_PATH.relative_to(BASE_DIR)}\n"
        f"  {XGB_PATH.relative_to(BASE_DIR)}\n"
        f"  {TEST_PATH.relative_to(BASE_DIR)}\n\n"
        "Run the app from inside the project folder so these relative paths resolve."
    )
    st.stop()
 
# Optional files — the app must never crash if these are absent.
y_test = None
_y_raw = load_optional_csv(YTEST_PATH)
if _y_raw is not None and not _y_raw.empty:
    y_test = _y_raw.squeeze()
 
train_df = load_optional_csv(TRAIN_PATH)
model_comparison_csv = load_optional_csv(MODEL_COMPARISON_PATH)
rf_importance_csv = load_optional_csv(RF_IMPORTANCE_PATH)
xgb_importance_csv = load_optional_csv(XGB_IMPORTANCE_PATH)
 
FEATURES = list(test_df.columns)
 
# ============================================================
# HELPERS
# ============================================================
 
def model_class_order(model, is_xgb):
    """Return this model's predict_proba column order, mapped to project labels 1/2/3."""
    classes = list(getattr(model, "classes_", [0, 1, 2] if is_xgb else [1, 2, 3]))
    if is_xgb:
        return [int(c) + 1 for c in classes]
    return [int(c) for c in classes]
 
def ordered_probs(model, X, is_xgb):
    """predict_proba re-ordered into fixed [Normal, Suspect, Pathological] order."""
    raw = model.predict_proba(X)[0]
    order = model_class_order(model, is_xgb)
    out = np.zeros(3)
    for val, label in zip(raw, order):
        out[label - 1] = val
    return out
 
def infer_single(model, X, is_xgb):
    """Returns (predicted_label 1/2/3, probs ordered [Normal,Suspect,Pathological], error|None)."""
    try:
        p = ordered_probs(model, X, is_xgb)
        predicted = STATE_ORDER[int(np.argmax(p))]
        return predicted, p, None
    except Exception as e:
        return None, None, str(e)
 
def state_name(x):
    return CLASS_NAMES.get(int(x), str(x))
 
def risk_score(probs_ordered):
    # Demonstration-only weighted probability score (not clinically validated).
    return float(probs_ordered[1] * 50 + probs_ordered[2] * 100)
 
def plotly_dark_layout(fig, height=350):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9e3f0", family="Inter"), margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b7c5d8")),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.09)", zerolinecolor="rgba(148,163,184,0.09)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.09)", zerolinecolor="rgba(148,163,184,0.09)")
    return fig
 
def risk_gauge(score, height=250):
    bar_color = "#38d39f" if score < 25 else ("#f7b84b" if score < 60 else "#ff5c69")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " /100", "font": {"size": 32, "color": "#f8fbff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5a78", "tickfont": {"color": "#8fa1bb"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)", "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(56,211,159,0.12)"},
                {"range": [25, 60], "color": "rgba(247,184,75,0.12)"},
                {"range": [60, 100], "color": "rgba(255,92,105,0.12)"},
            ],
        },
    ))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d9e3f0", family="Inter"), margin=dict(l=25, r=25, t=15, b=10))
    return fig
 
def probability_bar(probs_ordered, height=280, title="Model probability distribution"):
    fig = go.Figure(go.Bar(
        x=probs_ordered * 100, y=["Normal", "Suspect", "Pathological"], orientation="h",
        text=[f"{v*100:.1f}%" for v in probs_ordered], textposition="outside",
        marker=dict(color=["#38d39f", "#f7b84b", "#ff5c69"], line=dict(width=0)),
    ))
    fig.update_layout(title=title, xaxis=dict(range=[0, 108], title="Probability (%)"), yaxis=dict(title=""), showlegend=False)
    return plotly_dark_layout(fig, height)
 
def annotation_flag(feature_name):
    if feature_name in ANNOTATION_COLUMNS:
        return '<span class="feat-flag">ANNOTATION · LEAKAGE RISK</span>'
    if feature_name in METADATA_COLUMNS:
        return '<span class="feat-flag">SEGMENT METADATA</span>'
    return ""
 
def combined_importance():
    """Average of RF + XGB normalized feature importances, as a Series indexed by FEATURES."""
    rf_imp = getattr(rf_model, "feature_importances_", None)
    xgb_imp = getattr(xgb_model, "feature_importances_", None)
    parts = []
    if rf_imp is not None and len(rf_imp) == len(FEATURES):
        s = np.array(rf_imp, dtype=float)
        if s.sum() > 0:
            parts.append(s / s.sum())
    if xgb_imp is not None and len(xgb_imp) == len(FEATURES):
        s = np.array(xgb_imp, dtype=float)
        if s.sum() > 0:
            parts.append(s / s.sum())
    if not parts:
        return pd.Series(np.zeros(len(FEATURES)), index=FEATURES)
    avg = np.mean(parts, axis=0)
    return pd.Series(avg, index=FEATURES)
 
FEATURE_GUIDE = {
    "LB": "FHR baseline — the fetal heart rate baseline value, in beats per minute.",
    "AC": "Number of fetal heart rate accelerations, normalized per second of the recorded segment.",
    "FM": "Number of fetal movements detected, normalized per second of the recorded segment.",
    "UC": "Number of uterine contractions, normalized per second of the recorded segment.",
    "DL": "Number of light (mild) decelerations, normalized per second of the recorded segment.",
    "DS": "Number of severe decelerations, normalized per second of the recorded segment.",
    "DP": "Number of prolonged decelerations, normalized per second of the recorded segment.",
    "DR": "Number of repetitive decelerations recorded in the segment (constant / near-constant in this dataset).",
    "AC.1": "Alternate/repeated encoding of the accelerations count column from the source spreadsheet.",
    "FM.1": "Alternate/repeated encoding of the fetal movements column from the source spreadsheet.",
    "UC.1": "Alternate/repeated encoding of the uterine contractions column from the source spreadsheet.",
    "DL.1": "Alternate/repeated encoding of the light decelerations column from the source spreadsheet.",
    "DS.1": "Alternate/repeated encoding of the severe decelerations column from the source spreadsheet.",
    "DP.1": "Alternate/repeated encoding of the prolonged decelerations column from the source spreadsheet.",
    "ASTV": "Percentage of time with abnormal short-term variability in the fetal heart rate.",
    "MSTV": "Mean value of short-term variability of the fetal heart rate.",
    "ALTV": "Percentage of time with abnormal long-term variability in the fetal heart rate.",
    "MLTV": "Mean value of long-term variability of the fetal heart rate.",
    "Width": "Width of the FHR histogram (spread between max and min recorded values).",
    "Min": "Minimum value of the FHR histogram.",
    "Max": "Maximum value of the FHR histogram.",
    "Nmax": "Number of histogram peaks (local maxima) in the FHR distribution.",
    "Nzeros": "Number of histogram zero crossings/gaps in the FHR distribution.",
    "Mode": "Most frequently occurring value (mode) of the FHR histogram.",
    "Mean": "Mean value of the FHR histogram.",
    "Median": "Median value of the FHR histogram.",
    "Variance": "Variance of the FHR histogram, describing signal spread.",
    "Tendency": "Overall histogram tendency — a coded indicator of skew direction (left/symmetric/right).",
    "b": "Recording segment start instant (seconds) — session metadata, not a physiological signal.",
    "e": "Recording segment end instant (seconds) — session metadata, not a physiological signal.",
    "A": "One-hot indicator: calm-sleep morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "B": "One-hot indicator: REM-sleep morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "C": "One-hot indicator: calm-vigilance morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "D": "One-hot indicator: active-vigilance morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "E": "One-hot indicator: shift-pattern morphologic classification (FIGO/SisPorto CLASS annotation).",
    "AD": "One-hot indicator: accelerative/decelerative morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "DE": "One-hot indicator: decelerative pattern with vagal reaction (FIGO/SisPorto CLASS annotation).",
    "LD": "One-hot indicator: largely decelerative morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "FS": "One-hot indicator: flat-sinusoidal morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "SUSP": "One-hot indicator: suspect morphologic pattern (FIGO/SisPorto CLASS annotation).",
    "CLASS": "The 10-class FIGO/SisPorto morphologic pattern code itself, encoded as a single numeric column.",
}
 
# ============================================================
# SIDEBAR
# ============================================================
 
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">🫀 FetoGuard AI</div>
        <div class="brand-sub">Dual-model CTG intelligence</div>
    </div>
    """, unsafe_allow_html=True)
 
    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Live Assessment",
            "What-If Analysis",
            "Batch Triage",
            "Model Arena",
            "Explainability",
            "Dataset Analytics",
            "Feature Guide",
        ],
    )
 
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown('<div class="status-line"><span class="status-dot"></span> Random Forest loaded</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-line"><span class="status-dot"></span> XGBoost loaded</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-line"><span class="status-dot"></span> Dual-model inference ready</div>', unsafe_allow_html=True)
    if y_test is None:
        st.markdown('<div class="status-line"><span class="status-dot status-dot-warn"></span> y_test.csv not found — confusion matrix disabled</div>', unsafe_allow_html=True)
    if model_comparison_csv is None:
        st.markdown('<div class="status-line"><span class="status-dot status-dot-warn"></span> model_comparison.csv not found — using in-app metrics</div>', unsafe_allow_html=True)
    st.caption(f"Processed test samples · {len(test_df)}")
    st.caption(f"Feature columns · {len(FEATURES)}")
    st.caption("No retraining performed in-app")
 
# ============================================================
# OVERVIEW
# ============================================================
 
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-status"><span class="dot"></span> SYSTEM ONLINE</div>
        <div class="hero-kicker">AI • Cardiotocography • Dual-Model Decision Support</div>
        <div class="hero-title">AI-Powered <span>CTG Intelligence</span></div>
        <div class="hero-sub">
            FetoGuard AI runs Random Forest and XGBoost side by side on the same processed
            CTG feature profile, checks whether they agree, and blends their probabilities
            into a single risk-aware Dual Model verdict — with full explainability at every step.
        </div>
        <div class="pipeline-wrap">
            <div class="pipe-card"><div class="pipe-num">01</div><div class="pipe-label">CTG Features</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-card"><div class="pipe-num">02</div><div class="pipe-label">RF + XGBoost</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-card"><div class="pipe-num">03</div><div class="pipe-label">Agreement Check</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-card"><div class="pipe-num">04</div><div class="pipe-label">Dual Model Blend</div></div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-card"><div class="pipe-num">05</div><div class="pipe-label">Risk + Explanation</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="notice">
    <b>Prototype notice:</b> This hackathon system provides AI-assisted fetal-state
    classification for demonstration and research. It is a decision-support prototype,
    not a clinical diagnostic device — model outputs must not replace qualified
    professional assessment.
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown(f"""
    <div class="notice-warn">
    <b>⚠ Data integrity note:</b> {len(ANNOTATION_COLUMNS)} of the {len(FEATURES)} feature
    columns ({", ".join(sorted(ANNOTATION_COLUMNS))}) are FIGO/SisPorto morphologic-pattern
    <i>annotations</i> rather than independent CTG signal measurements, and are strongly
    correlated with the target label. This is very likely why held-out accuracy exceeds 99%
    for both models. <code>notebook/05_leakage_check.ipynb</code> already flags these exact
    columns for removal — see the Feature Guide tab for details on every column.
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="section-kicker">Quick stats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System at a glance</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="stat-card"><div class="stat-label">Output classes</div><div class="stat-value">3 states</div><div class="stat-note">Normal · Suspect · Pathological</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Feature columns</div><div class="stat-value">{len(FEATURES)}</div><div class="stat-note">Processed CTG-derived inputs</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Held-out test set</div><div class="stat-value">{len(test_df)}</div><div class="stat-note">Records used for evaluation</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-card"><div class="stat-label">Engines</div><div class="stat-value">2 models</div><div class="stat-note">Random Forest + XGBoost, blended</div></div>', unsafe_allow_html=True)
 
    st.write("")
    st.markdown('<div class="section-kicker">Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why FetoGuard?</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="glass-card"><div class="why-num">01</div><div class="why-title">Dual-Model Consensus</div>
        <div class="why-desc">Two independently trained models vote on every sample. Agreement
        raises confidence in the result; disagreement is surfaced explicitly rather than hidden
        behind a single number.</div></div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="glass-card"><div class="why-num">02</div><div class="why-title">Risk-Aware Predictions</div>
        <div class="why-desc">Every prediction ships with a full probability profile and a
        prototype 0–100 risk signal derived from blended class probabilities.</div></div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="glass-card"><div class="why-num">03</div><div class="why-title">Explainable + Interactive</div>
        <div class="why-desc">Real feature-importance rankings from both models, plus a
        What-If Analysis tool to see how changing a feature value shifts the verdict live.</div></div>
        """, unsafe_allow_html=True)
 
# ============================================================
# LIVE ASSESSMENT — DUAL MODEL
# ============================================================
 
elif page == "Live Assessment":
    st.markdown('<div class="section-kicker">Primary workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live CTG Assessment · Dual Model</div>', unsafe_allow_html=True)
    st.write("Random Forest and XGBoost both evaluate the same processed CTG feature profile.")
 
    input_mode = st.radio("INPUT SOURCE", ["Select test sample", "Upload processed feature CSV"], horizontal=True)
 
    selected = None
    upload_error = None
 
    if input_mode == "Select test sample":
        sample_no = st.number_input("TEST SAMPLE", min_value=1, max_value=len(test_df), value=1, step=1)
        selected = test_df.iloc[[int(sample_no) - 1]].copy()
    else:
        uploaded_single = st.file_uploader("Upload a single-row processed feature CSV", type=["csv"], key="single_upload")
        if uploaded_single is not None:
            try:
                row = pd.read_csv(uploaded_single)
                missing = [f for f in FEATURES if f not in row.columns]
                extra = [c for c in row.columns if c not in FEATURES]
                if row.empty:
                    upload_error = "The uploaded file has no rows."
                elif missing:
                    upload_error = f"Missing {len(missing)} required column(s): {', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}"
                else:
                    if extra:
                        st.caption(f"Ignoring {len(extra)} extra column(s) not used by the models.")
                    selected = row[FEATURES].iloc[[0]].copy()
                    numeric_check = selected.apply(pd.to_numeric, errors="coerce")
                    if numeric_check.isnull().any(axis=1).iloc[0]:
                        upload_error = "The selected row contains missing/invalid numeric values."
                        selected = None
                    else:
                        selected = numeric_check
            except Exception as e:
                upload_error = f"Could not read the uploaded CSV: {e}"
 
    if upload_error:
        st.error(upload_error)
 
    if selected is not None:
        with st.expander("View processed feature values"):
            st.dataframe(selected.T.rename(columns={selected.index[0]: "Value"}), use_container_width=True)
 
        run = st.button("⚡ RUN FETOGUARD ANALYSIS →", type="primary", use_container_width=True)
 
        if run:
            with st.status("Analyzing CTG profile...", expanded=True) as status:
                st.write("✓ Feature vector loaded")
                rf_pred, rf_probs, rf_err = infer_single(rf_model, selected, is_xgb=False)
                xgb_pred, xgb_probs, xgb_err = infer_single(xgb_model, selected, is_xgb=True)
 
                if rf_err or xgb_err:
                    status.update(label="Analysis failed", state="error")
                    st.error("FetoGuard could not complete inference on this input.")
                    if rf_err:
                        st.code(f"Random Forest error: {rf_err}")
                    if xgb_err:
                        st.code(f"XGBoost error: {xgb_err}")
                    st.stop()
 
                st.write("✓ Random Forest inference complete")
                st.write("✓ XGBoost inference complete")
 
                dual_probs = (rf_probs + xgb_probs) / 2.0
                dual_pred = STATE_ORDER[int(np.argmax(dual_probs))]
                dual_conf = float(np.max(dual_probs))
                dual_score = risk_score(dual_probs)
 
                st.write("✓ Confidence calculated")
                st.write("✓ Dual-model blend + explanation generated")
                status.update(label="Analysis complete", state="complete")
 
            rf_conf = float(np.max(rf_probs))
            xgb_conf = float(np.max(xgb_probs))
            agree = (rf_pred == xgb_pred)
 
            st.write("")
            st.markdown('<div class="section-kicker">Per-model results</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Individual Model Predictions</div>', unsafe_allow_html=True)
 
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"""
                <div class="model-card model-card-rf">
                    <div class="model-card-head">
                        <div>🌲</div>
                        <div>
                            <div class="model-card-title">Random Forest</div>
                            <div class="model-card-sub">Ensemble · saved classifier</div>
                        </div>
                    </div>
                    <div class="model-card-state" style="color:{CLASS_COLORS[rf_pred]};">{CLASS_ICONS[rf_pred]} {state_name(rf_pred).upper()}</div>
                    <div class="model-card-conf">{rf_conf*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="model-card model-card-xgb">
                    <div class="model-card-head">
                        <div>⚡</div>
                        <div>
                            <div class="model-card-title">XGBoost</div>
                            <div class="model-card-sub">Gradient boosted · saved classifier</div>
                        </div>
                    </div>
                    <div class="model-card-state" style="color:{CLASS_COLORS[xgb_pred]};">{CLASS_ICONS[xgb_pred]} {state_name(xgb_pred).upper()}</div>
                    <div class="model-card-conf">{xgb_conf*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
 
            if agree:
                st.markdown(
                    f'<div class="agree-banner agree-yes">🤝 MODELS AGREE'
                    f'<div class="agree-sub">Both Random Forest and XGBoost independently predict {state_name(rf_pred)}.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="agree-banner agree-no">⚠ MODELS DISAGREE'
                    f'<div class="agree-sub">Random Forest predicts {state_name(rf_pred)}, XGBoost predicts {state_name(xgb_pred)}. '
                    f'Treat this case as higher priority for manual review.</div></div>',
                    unsafe_allow_html=True,
                )
 
            st.write("")
            st.markdown('<div class="section-kicker">Blended verdict</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🏆 Dual Model — Final Result</div>', unsafe_allow_html=True)
 
            st.markdown(f"""
            <div class="dual-card">
                <div class="dual-kicker">Dual Model · average of RF + XGBoost probabilities</div>
                <div class="dual-state" style="color:{CLASS_COLORS[dual_pred]};">{CLASS_ICONS[dual_pred]} {state_name(dual_pred).upper()}</div>
                <div class="dual-conf">{dual_conf*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
 
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Final state</div><div class="stat-value">{CLASS_ICONS[dual_pred]} {state_name(dual_pred)}</div><div class="stat-note">Dual Model verdict</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Confidence</div><div class="stat-value">{dual_conf*100:.1f}%</div><div class="stat-note">Blended probability</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Risk signal</div><div class="stat-value">{dual_score:.1f}/100</div><div class="stat-note">Prototype-derived score</div></div>', unsafe_allow_html=True)
            with m4:
                agree_label = "✓ Agree" if agree else "⚠ Disagree"
                st.markdown(f'<div class="stat-card"><div class="stat-label">Model agreement</div><div class="stat-value">{agree_label}</div><div class="stat-note">RF vs XGBoost</div></div>', unsafe_allow_html=True)
 
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("#### Dual Model probability profile")
                st.plotly_chart(probability_bar(dual_probs, 300, "Blended (RF + XGBoost average)"), use_container_width=True)
 
                st.markdown("#### Model-derived risk signal")
                st.plotly_chart(risk_gauge(dual_score, 250), use_container_width=True)
                st.caption("This is a prototype score derived from blended class probabilities. It is not a clinically validated hypoxia index.")
 
            with right:
                st.markdown("#### Individual probability profiles")
                st.caption("Random Forest")
                st.plotly_chart(probability_bar(rf_probs, 190, ""), use_container_width=True)
                st.caption("XGBoost")
                st.plotly_chart(probability_bar(xgb_probs, 190, ""), use_container_width=True)
 
            st.write("")
            st.markdown("#### What the models see (combined importance)")
            imp = combined_importance().sort_values(ascending=False).head(8)
            colors = ["#ffb974" if f in ANNOTATION_COLUMNS else "#7779ff" for f in imp.index]
            fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=colors)))
            fig2.update_layout(title="Top features — averaged RF + XGBoost importance", xaxis_title="Normalized importance", yaxis_title="")
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(plotly_dark_layout(fig2, 320), use_container_width=True)
            if any(f in ANNOTATION_COLUMNS for f in imp.index):
                st.caption("🟠 Orange bars are annotation-derived columns, not raw CTG signal features — see the Overview data-integrity note.")
            st.caption("Feature importance describes model reliance, not biological causation. This is an AI-assisted classification, not a medical diagnosis.")
 
# ============================================================
# WHAT-IF ANALYSIS
# ============================================================
 
elif page == "What-If Analysis":
    st.markdown('<div class="section-kicker">Interactive explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 What-If Analysis</div>', unsafe_allow_html=True)
    st.write("Start from a real test sample, nudge its most influential features, and watch both models' verdicts update live.")
 
    sample_no = st.number_input("BASE TEST SAMPLE", min_value=1, max_value=len(test_df), value=1, step=1, key="whatif_sample")
    base_row = test_df.iloc[[int(sample_no) - 1]].copy()
 
    n_features = st.slider("Number of adjustable features", min_value=3, max_value=12, value=6)
    imp = combined_importance().sort_values(ascending=False)
    top_features = list(imp.head(n_features).index)
 
    range_source = train_df if train_df is not None else test_df
 
    st.markdown("#### Adjust feature values")
    what_if_row = base_row.copy()
    cols = st.columns(2)
    for i, feat in enumerate(top_features):
        col = cols[i % 2]
        col_min = float(range_source[feat].min())
        col_max = float(range_source[feat].max())
        base_val = float(base_row[feat].iloc[0])
        if col_min == col_max:
            col_max = col_min + 1.0
        flag = " ⚠ annotation column" if feat in ANNOTATION_COLUMNS else ""
        with col:
            new_val = st.slider(
                f"{feat}{flag}",
                min_value=col_min, max_value=col_max, value=base_val,
                key=f"whatif_{feat}",
            )
            what_if_row[feat] = new_val
 
    if st.button("↺ Reset sliders to sample values", use_container_width=False):
        for feat in top_features:
            st.session_state.pop(f"whatif_{feat}", None)
        st.rerun()
 
    base_rf_pred, base_rf_probs, e1 = infer_single(rf_model, base_row, is_xgb=False)
    base_xgb_pred, base_xgb_probs, e2 = infer_single(xgb_model, base_row, is_xgb=True)
    new_rf_pred, new_rf_probs, e3 = infer_single(rf_model, what_if_row, is_xgb=False)
    new_xgb_pred, new_xgb_probs, e4 = infer_single(xgb_model, what_if_row, is_xgb=True)
 
    if any([e1, e2, e3, e4]):
        st.error("Could not compute the What-If prediction.")
        st.code(next(e for e in [e1, e2, e3, e4] if e))
    else:
        base_dual = (base_rf_probs + base_xgb_probs) / 2.0
        new_dual = (new_rf_probs + new_xgb_probs) / 2.0
        base_dual_pred = STATE_ORDER[int(np.argmax(base_dual))]
        new_dual_pred = STATE_ORDER[int(np.argmax(new_dual))]
        base_dual_conf = float(np.max(base_dual))
        new_dual_conf = float(np.max(new_dual))
        base_risk = risk_score(base_dual)
        new_risk = risk_score(new_dual)
 
        st.write("")
        st.markdown('<div class="section-kicker">Live comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Baseline vs. What-If (Dual Model)</div>', unsafe_allow_html=True)
 
        b1, b2 = st.columns(2)
        with b1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">BASELINE — sample #{int(sample_no)}</div>
                <div class="stat-value" style="color:{CLASS_COLORS[base_dual_pred]};">{CLASS_ICONS[base_dual_pred]} {state_name(base_dual_pred)}</div>
                <div class="stat-note">{base_dual_conf*100:.1f}% confidence · risk {base_risk:.1f}/100</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            delta = new_risk - base_risk
            delta_class = "delta-flat"
            if delta > 0.5:
                delta_class = "delta-up"
            elif delta < -0.5:
                delta_class = "delta-down"
            changed = " (state changed)" if new_dual_pred != base_dual_pred else ""
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-label">WHAT-IF</div>
                <div class="stat-value" style="color:{CLASS_COLORS[new_dual_pred]};">{CLASS_ICONS[new_dual_pred]} {state_name(new_dual_pred)}
                <span class="delta-chip {delta_class}">{'+' if delta>=0 else ''}{delta:.1f} risk</span></div>
                <div class="stat-note">{new_dual_conf*100:.1f}% confidence · risk {new_risk:.1f}/100{changed}</div>
            </div>
            """, unsafe_allow_html=True)
 
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(probability_bar(base_dual, 260, "Baseline probability profile"), use_container_width=True)
        with c2:
            st.plotly_chart(probability_bar(new_dual, 260, "What-If probability profile"), use_container_width=True)
 
        st.caption(
            "What-If Analysis perturbs a real held-out sample's feature values and re-runs both "
            "saved models — nothing here retrains or modifies the models. Useful for stress-testing "
            "how sensitive the Dual Model verdict is to a given feature."
        )
 
# ============================================================
# BATCH TRIAGE
# ============================================================
 
elif page == "Batch Triage":
    st.markdown('<div class="section-kicker">Scale the workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Batch Patient Triage</div>', unsafe_allow_html=True)
    st.write("Upload a CSV containing the same processed feature columns used by the saved models. Both models run on every row.")
 
    uploaded = st.file_uploader("Upload processed CTG CSV", type=["csv"])
 
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:
            batch = None
            st.error("Could not read the uploaded file as a CSV.")
            st.code(str(e))
 
        if batch is not None:
            if batch.empty:
                st.error("The uploaded file has no rows.")
            else:
                missing = [f for f in FEATURES if f not in batch.columns]
                if missing:
                    st.error(f"Missing {len(missing)} required feature column(s).")
                    st.code(", ".join(missing))
                else:
                    X = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                    bad_rows = X.isnull().any(axis=1)
                    if bad_rows.any():
                        st.warning(f"{int(bad_rows.sum())} row(s) contain missing or non-numeric values and will be excluded.")
                        batch = batch.loc[~bad_rows].reset_index(drop=True)
                        X = X.loc[~bad_rows].reset_index(drop=True)
 
                    if X.empty:
                        st.error("No valid rows remain after validation.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f'<div class="stat-card"><div class="stat-label">Records detected</div><div class="stat-value">{len(batch)}</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="stat-card"><div class="stat-label">Features validated</div><div class="stat-value">{len(FEATURES)}/{len(FEATURES)}</div></div>', unsafe_allow_html=True)
                        with c3:
                            st.markdown('<div class="stat-card"><div class="stat-label">Engines</div><div class="stat-value">RF + XGBoost</div></div>', unsafe_allow_html=True)
 
                        st.write("")
                        if st.button("Analyze Batch →", type="primary", use_container_width=True):
                            try:
                                rf_raw = rf_model.predict_proba(X)
                                xgb_raw = xgb_model.predict_proba(X)
                                rf_order = model_class_order(rf_model, is_xgb=False)
                                xgb_order = model_class_order(xgb_model, is_xgb=True)
 
                                rf_ordered = np.zeros((len(X), 3))
                                for j, lbl in enumerate(rf_order):
                                    rf_ordered[:, lbl - 1] = rf_raw[:, j]
                                xgb_ordered = np.zeros((len(X), 3))
                                for j, lbl in enumerate(xgb_order):
                                    xgb_ordered[:, lbl - 1] = xgb_raw[:, j]
 
                                dual_ordered = (rf_ordered + xgb_ordered) / 2.0
                                rf_pred = np.array(STATE_ORDER)[np.argmax(rf_ordered, axis=1)]
                                xgb_pred = np.array(STATE_ORDER)[np.argmax(xgb_ordered, axis=1)]
                                dual_pred = np.array(STATE_ORDER)[np.argmax(dual_ordered, axis=1)]
 
                                results = batch.copy()
                                results["RF_Predicted_State"] = [state_name(x) for x in rf_pred]
                                results["XGB_Predicted_State"] = [state_name(x) for x in xgb_pred]
                                results["Models_Agree"] = rf_pred == xgb_pred
                                results["Dual_Predicted_State"] = [state_name(x) for x in dual_pred]
                                results["Dual_Normal_Probability"] = dual_ordered[:, 0]
                                results["Dual_Suspect_Probability"] = dual_ordered[:, 1]
                                results["Dual_Pathological_Probability"] = dual_ordered[:, 2]
                                results["Dual_Risk_Signal"] = dual_ordered[:, 1] * 50 + dual_ordered[:, 2] * 100
 
                                n_disagree = int((~results["Models_Agree"]).sum())
                                st.success(f"Analyzed {len(results)} records with both models. {n_disagree} record(s) show model disagreement.")
                                st.dataframe(results, use_container_width=True, height=430)
                                st.download_button(
                                    "📥 DOWNLOAD PREDICTIONS",
                                    results.to_csv(index=False).encode("utf-8"),
                                    "fetoguard_dual_model_predictions.csv",
                                    "text/csv",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error("Batch inference failed.")
                                st.code(str(e))
 
# ============================================================
# MODEL ARENA
# ============================================================
 
elif page == "Model Arena":
    st.markdown('<div class="section-kicker">Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Model Arena</div>', unsafe_allow_html=True)
    st.write("Two ensemble learners. One held-out test set.")
 
    # Prefer numbers from results/model_comparison.csv when present; fall back gracefully.
    fallback = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [0.9952830188679245, 0.9929245283018868],
        "Macro F1": [0.9932458242178005, 0.9897960522001443],
    })
    comparison = fallback
    if model_comparison_csv is not None:
        cols = {c.lower().replace(" ", ""): c for c in model_comparison_csv.columns}
        if "model" in cols and "accuracy" in cols:
            f1_col = cols.get("macrof1") or cols.get("macro_f1")
            try:
                comparison = pd.DataFrame({
                    "Model": model_comparison_csv[cols["model"]],
                    "Accuracy": model_comparison_csv[cols["accuracy"]],
                    "Macro F1": model_comparison_csv[f1_col] if f1_col else fallback["Macro F1"],
                })
            except Exception:
                comparison = fallback
 
    rf_row = comparison[comparison["Model"].str.contains("Random", case=False, na=False)]
    xgb_row = comparison[comparison["Model"].str.contains("XGB", case=False, na=False)]
    rf_acc = float(rf_row["Accuracy"].iloc[0]) if not rf_row.empty else fallback["Accuracy"][0]
    rf_f1 = float(rf_row["Macro F1"].iloc[0]) if not rf_row.empty else fallback["Macro F1"][0]
    xgb_acc = float(xgb_row["Accuracy"].iloc[0]) if not xgb_row.empty else fallback["Accuracy"][1]
    xgb_f1 = float(xgb_row["Macro F1"].iloc[0]) if not xgb_row.empty else fallback["Macro F1"][1]
    rf_wins = rf_acc >= xgb_acc
 
    c1, vs, c2 = st.columns([1, 0.15, 1])
    with c1:
        st.markdown(f"""
        <div class="arena-card {'arena-winner' if rf_wins else ''}">
            {'<div class="arena-badge">WINNER</div>' if rf_wins else ''}
            <div class="arena-name">🌲 Random Forest</div>
            <div class="arena-metric-row">
                <div><div class="arena-metric-val">{rf_acc*100:.2f}%</div><div class="arena-metric-label">Accuracy</div></div>
                <div><div class="arena-metric-val">{rf_f1*100:.2f}%</div><div class="arena-metric-label">Macro F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with vs:
        st.markdown('<div class="vs-badge" style="padding-top:38px;">VS</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="arena-card {'arena-winner' if not rf_wins else ''}">
            {'<div class="arena-badge">WINNER</div>' if not rf_wins else ''}
            <div class="arena-name">⚡ XGBoost</div>
            <div class="arena-metric-row">
                <div><div class="arena-metric-val">{xgb_acc*100:.2f}%</div><div class="arena-metric-label">Accuracy</div></div>
                <div><div class="arena-metric-val">{xgb_f1*100:.2f}%</div><div class="arena-metric-label">Macro F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.write("")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=["Random Forest", "XGBoost"], y=[rf_acc*100, xgb_acc*100],
                          text=[f"{rf_acc*100:.2f}%", f"{xgb_acc*100:.2f}%"], textposition="outside", marker_color="#7779ff"))
    fig.add_trace(go.Bar(name="Macro F1", x=["Random Forest", "XGBoost"], y=[rf_f1*100, xgb_f1*100],
                          text=[f"{rf_f1*100:.2f}%", f"{xgb_f1*100:.2f}%"], textposition="outside", marker_color="#38d39f"))
    fig.update_layout(title="Held-out test performance", barmode="group", yaxis=dict(range=[95, 100], title="Score (%)"))
    st.plotly_chart(plotly_dark_layout(fig, 420), use_container_width=True)
 
    if y_test is not None:
        chosen = st.selectbox("Confusion matrix", ["Random Forest", "XGBoost", "Dual Model"])
        try:
            if chosen == "Random Forest":
                pred = np.array(STATE_ORDER)[np.argmax(np.stack([ordered_probs(rf_model, test_df.iloc[[i]], False) for i in range(len(test_df))]), axis=1)]
            elif chosen == "XGBoost":
                pred = np.array(STATE_ORDER)[np.argmax(np.stack([ordered_probs(xgb_model, test_df.iloc[[i]], True) for i in range(len(test_df))]), axis=1)]
            else:
                rf_all = rf_model.predict_proba(test_df)
                xgb_all = xgb_model.predict_proba(test_df)
                rf_order = model_class_order(rf_model, False)
                xgb_order = model_class_order(xgb_model, True)
                rf_ord = np.zeros((len(test_df), 3))
                for j, lbl in enumerate(rf_order):
                    rf_ord[:, lbl - 1] = rf_all[:, j]
                xgb_ord = np.zeros((len(test_df), 3))
                for j, lbl in enumerate(xgb_order):
                    xgb_ord[:, lbl - 1] = xgb_all[:, j]
                dual_ord = (rf_ord + xgb_ord) / 2.0
                pred = np.array(STATE_ORDER)[np.argmax(dual_ord, axis=1)]
 
            cm = confusion_matrix(y_test, pred, labels=STATE_ORDER)
            labels = ["Normal", "Suspect", "Pathological"]
            fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}", colorscale="Viridis", showscale=True))
            fig.update_layout(title=f"Confusion Matrix — {chosen}", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(plotly_dark_layout(fig, 420), use_container_width=True)
        except Exception as e:
            st.error("Could not compute confusion matrix.")
            st.code(str(e))
    else:
        st.info("Ground-truth test labels (y_test.csv) were not found, so a confusion matrix cannot be computed.")
 
    st.markdown("""
    <div class="notice-warn">
    <b>⚠ Read these numbers with caution:</b> both models were evaluated using a feature set
    that still includes FIGO/SisPorto annotation columns (see Overview / Feature Guide). The
    near-99% accuracy is likely inflated by that leakage rather than reflecting real-world
    performance on raw CTG signal features alone.
    </div>
    """, unsafe_allow_html=True)
 
# ============================================================
# EXPLAINABILITY
# ============================================================
 
elif page == "Explainability":
    st.markdown('<div class="section-kicker">Model transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Why did the models predict this?</div>', unsafe_allow_html=True)
    st.write("Feature importance shows which processed variables each model relied on most. It should not be interpreted as proof of causation.")
 
    tab_rf, tab_xgb, tab_combined = st.tabs(["🌲 Random Forest", "⚡ XGBoost", "🤝 Combined"])
 
    def importance_panel(model, label):
        importance = getattr(model, "feature_importances_", None)
        if importance is None:
            st.info(f"Feature importance is not exposed by the {label} model.")
            return
        imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importance}).sort_values("Importance", ascending=False)
        top = imp_df.head(15)
        colors = ["#ffb974" if f in ANNOTATION_COLUMNS else "#7779ff" for f in top["Feature"]]
        fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h", marker_color=colors))
        fig.update_layout(title=f"Top 15 features — {label}", xaxis_title="Importance", yaxis_title="")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_dark_layout(fig, 520), use_container_width=True)
        st.caption("🟠 Orange bars mark annotation-derived columns rather than raw CTG signal features.")
        st.dataframe(top.reset_index(drop=True), use_container_width=True, hide_index=True)
 
    with tab_rf:
        importance_panel(rf_model, "Random Forest")
    with tab_xgb:
        importance_panel(xgb_model, "XGBoost")
    with tab_combined:
        imp = combined_importance().sort_values(ascending=False).head(15)
        colors = ["#ffb974" if f in ANNOTATION_COLUMNS else "#72e6d7" for f in imp.index]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=colors))
        fig.update_layout(title="Top 15 features — averaged RF + XGBoost", xaxis_title="Normalized importance", yaxis_title="")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_dark_layout(fig, 520), use_container_width=True)
        st.caption("Average of each model's normalized feature_importances_. Used to pick sliders on the What-If Analysis page.")
 
    st.caption("Model-derived feature importance. Importance indicates model reliance, not biological causation.")
 
# ============================================================
# DATASET ANALYTICS
# ============================================================
 
elif page == "Dataset Analytics":
    st.markdown('<div class="section-kicker">Understand the input</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Processed Dataset Analytics</div>', unsafe_allow_html=True)
 
    x, y, z, w = st.columns(4)
    with x:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Test samples</div><div class="stat-value">{len(test_df)}</div><div class="stat-note">Held-out records</div></div>', unsafe_allow_html=True)
    with y:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Feature count</div><div class="stat-value">{len(FEATURES)}</div><div class="stat-note">Processed columns</div></div>', unsafe_allow_html=True)
    with z:
        train_n = len(train_df) if train_df is not None else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-label">Train samples</div><div class="stat-value">{train_n}</div><div class="stat-note">Used for the saved models</div></div>', unsafe_allow_html=True)
    with w:
        st.markdown('<div class="stat-card"><div class="stat-label">Pipeline</div><div class="stat-value">Frozen</div><div class="stat-note">No retraining in-app</div></div>', unsafe_allow_html=True)
 
    st.write("")
    if y_test is not None:
        st.markdown("#### Class distribution — held-out test set")
        counts = y_test.value_counts().sort_index()
        names = [state_name(i) for i in counts.index]
        colors = [CLASS_COLORS.get(int(i), "#7779ff") for i in counts.index]
        fig = go.Figure(go.Bar(x=names, y=counts.values, marker_color=colors, text=counts.values, textposition="outside"))
        fig.update_layout(title="Ground-truth label distribution (NSP)", xaxis_title="", yaxis_title="Records")
        st.plotly_chart(plotly_dark_layout(fig, 360), use_container_width=True)
        st.caption("The dataset is imbalanced toward Normal cases — this is why FetoGuard reports macro-averaged F1 alongside accuracy.")
    else:
        st.info("y_test.csv was not found, so class distribution cannot be shown.")
 
    st.write("")
    st.markdown("#### Feature distribution explorer")
    feature = st.selectbox("Explore a feature", FEATURES)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=test_df[feature], nbinsx=35, marker=dict(color="#6d70ff")))
    fig.update_layout(title=f"Distribution of {feature}", xaxis_title=feature, yaxis_title="Records")
    st.plotly_chart(plotly_dark_layout(fig, 400), use_container_width=True)
    if feature in ANNOTATION_COLUMNS:
        st.caption("⚠ This column is an annotation/leakage-risk feature, not a raw CTG signal measurement.")
 
# ============================================================
# FEATURE GUIDE
# ============================================================
 
elif page == "Feature Guide":
    st.markdown('<div class="section-kicker">Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Guide</div>', unsafe_allow_html=True)
    st.write("A searchable dictionary of every processed feature column used by the saved models.")
 
    query = st.text_input("Search features", placeholder="e.g. ASTV, deceleration, variability…")
 
    rows = [(f, FEATURE_GUIDE.get(f, "Processed feature column from the existing dataset.")) for f in FEATURES]
    if query:
        q = query.lower()
        rows = [r for r in rows if q in r[0].lower() or q in r[1].lower()]
 
    if not rows:
        st.info("No features match your search.")
    else:
        for f, desc in rows:
            flag = annotation_flag(f)
            st.markdown(f'<div class="feat-row"><span class="feat-tag">{f}</span>{flag}<div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)
 
    st.caption(
        "Descriptions follow the standard CTG / SisPorto feature definitions from the "
        "source dataset. Only documented, supportable descriptions are shown — no clinical "
        "thresholds or biological claims are invented here."
    )
 
# ============================================================
# FOOTER
# ============================================================
 
st.markdown("""
<div class="footer">
    🫀 <b>FetoGuard AI</b> · Dual-Model Explainable & Risk-Aware CTG Fetal-State Classification<br>
    Hackathon prototype · Not a clinical diagnostic device
</div>
""", unsafe_allow_html=True)
 




