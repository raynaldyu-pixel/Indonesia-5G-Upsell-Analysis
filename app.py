"""
The Booster Signal — Interactive Indonesian 5G Upsell Analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────────────────
# Page config & theme
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Booster Signal — 5G Upsell Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY   = "#0B2545"
TEAL   = "#1B998B"
C_RED    = "#E53935"
C_BLUE   = "#1E88E5"
C_LIGHT  = "#90CAF9"
C_GREEN  = "#43A047"
C_ORANGE = "#FB8C00"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7F9FB; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #EAF0F6 !important; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    .booster-badge {{
        display:inline-block; background:{TEAL}; color:white; padding:2px 10px;
        border-radius:12px; font-size:0.75rem; font-weight:600; margin-left:8px;
    }}
    
    /* ── 1. Code Block Background ── */
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"],
    section[data-testid="stSidebar"] [data-testid="stCodeBlock"] > div,
    section[data-testid="stSidebar"] pre, 
    section[data-testid="stSidebar"] code {{
        background-color: #262730 !important; 
        background: #262730 !important;
        border: none !important;
    }}

    /* ── 2. Number Input Text Boxes (FIXED FOR LIGHT THEME) ── */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"], 
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="base-input"] {{
        background-color: #262730 !important;
        border-color: #3E404F !important;
    }}
    /* Forces the exact text box to be dark grey instead of transparent */
    section[data-testid="stSidebar"] input {{
        background-color: #262730 !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
    }}
    
    /* ── 3. Number Input (+ / -) Buttons ── */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{
        background-color: #3E404F !important;
        border-color: #3E404F !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {{
        fill: white !important;
        color: white !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {{
        background-color: #525468 !important;
    }}
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = [
    'Customer_ID', 'churn', 'hnd_webcap',
    'totmou', 'totrev', 'avgmou', 'avgrev', 'rev_Mean',
    'ovrmou_Mean', 'ovrrev_Mean', 'change_mou', 'change_rev',
    'eqpdays', 'months', 'roam_Mean', 'drop_vce_Mean',
    'blck_vce_Mean', 'custcare_Mean',
]

NUMERIC_FEATS_BASE = [
    'totmou', 'totrev', 'avgmou', 'avgrev',
    'ovrmou_Mean', 'ovrrev_Mean', 'eqpdays', 'months',
    'roam_Mean', 'drop_vce_Mean', 'blck_vce_Mean', 'custcare_Mean',
    'data_hunger_velocity', 'rev_momentum',
    'booster_intensity', 'booster_freq_proxy', 'engagement_depth',
]

BOOSTER_FEATS = {'booster_intensity', 'booster_freq_proxy',
                  'data_hunger_velocity', 'ovrmou_Mean', 'ovrrev_Mean'}

# ──────────────────────────────────────────────────────────────────────────
# Path and Data
# ──────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent

CLIENT_CSV_PATH = APP_DIR / "data" / "Client.csv"
RECORD_CSV_PATH = APP_DIR / "data" / "Record.csv"
MODEL_PATH = APP_DIR / "model" / "5G_upsell_xgb_model.json"


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(_client_path_str, _record_path_str, _mtime_key):
    # _mtime_key busts the cache whenever the files on disk change
    client_df = pd.read_csv(_client_path_str)
    record_df = pd.read_csv(_record_path_str)
    df = pd.merge(client_df, record_df, on='Customer_ID', how='inner')
    return df


@st.cache_data(show_spinner=False)
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col in ['ovrmou_Mean', 'ovrrev_Mean', 'change_mou', 'change_rev', 'rev_Mean']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    rev_median = df['rev_Mean'].median() if 'rev_Mean' in df.columns else 0
    above_rev = (df['rev_Mean'] > rev_median) if 'rev_Mean' in df.columns else True

    df['Upsell_Target'] = (
        (df['ovrrev_Mean'] > 0) &
        (df['hnd_webcap'].isin(['WCMB', 'WC'])) &
        above_rev &
        (df['churn'] == 0)
    ).astype(int)

    df['booster_intensity'] = np.where(
        df.get('totrev', pd.Series(0, index=df.index)).fillna(0) > 0,
        df['ovrrev_Mean'] / df['totrev'].replace(0, np.nan), 0
    )
    df['booster_intensity'] = df['booster_intensity'].clip(0, 1).fillna(0)

    df['data_hunger_velocity'] = df['change_mou'].clip(lower=-500, upper=2000)
    df['rev_momentum'] = df['change_rev'].clip(lower=-100, upper=500)

    df['engagement_depth'] = (
        df.get('totmou', pd.Series(0, index=df.index)).fillna(0) /
        df.get('months', pd.Series(1, index=df.index)).fillna(1).replace(0, 1)
    )

    df['booster_freq_proxy'] = np.where(
        df.get('totmou', pd.Series(0, index=df.index)).fillna(0) > 0,
        df['ovrmou_Mean'] / df['totmou'].replace(0, np.nan), 0
    )
    df['booster_freq_proxy'] = df['booster_freq_proxy'].clip(0, 1).fillna(0)

    df.attrs['rev_median'] = rev_median
    return df


# ──────────────────────────────────────────────────────────────────────────
# Pre-trained model loading
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pretrained_model(_path_str, _mtime_key):
    path = Path(_path_str)
    suffix = path.suffix.lower()

    if suffix in (".pkl", ".joblib", ".pickle"):
        model = joblib.load(path)
    elif suffix in (".json", ".ubj"):
        model = XGBClassifier()
        model.load_model(str(path))
    else:
        raise ValueError(f"Unsupported model file type: {suffix}. "
                          f"Use .pkl / .joblib (sklearn dump) or .json / .ubj (native XGBoost).")

    feature_list = None
    if hasattr(model, "feature_names_in_"):
        feature_list = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            feature_list = model.get_booster().feature_names
        except Exception:
            feature_list = None

    return model, feature_list


def build_result_from_pretrained(model, feature_list, df, eval_fraction, seed):
    """Score the pre-trained model against a fresh evaluation slice of the
    current data so the rest of the app (ROI, decile lift, go-to-market)
    works exactly as if training had just happened here."""
    features = [f for f in feature_list if f in df.columns] if feature_list else \
        [f for f in NUMERIC_FEATS_BASE if f in df.columns]
    missing = [f for f in (feature_list or []) if f not in df.columns]

    X = df[features].fillna(df[features].median())
    y = df['Upsell_Target']

    if eval_fraction < 1.0:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=eval_fraction, random_state=seed, stratify=y
        )
    else:
        X_eval, y_eval = X, y

    y_prob = model.predict_proba(X_eval)[:, 1]

    return {
        'model': model, 'X_test': X_eval, 'y_test': y_eval, 'y_prob': y_prob,
        'features': features, 'missing_features': missing,
    }


def fmt_rp(val):
    return f"Rp {val:,.0f}"


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — data & financial baseline controls
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📡 The Booster Signal")
st.sidebar.caption("5G Upsell Targeting — Indonesian Prepaid Market")
st.sidebar.markdown("---")

st.sidebar.markdown("### Data & Model")
st.sidebar.code(f"data/{CLIENT_CSV_PATH.name}\ndata/{RECORD_CSV_PATH.name}\n"
                 f"model/{MODEL_PATH.name}", language=None)

# ── Load data ──────────────────────────────────────────────────────────
if not (CLIENT_CSV_PATH.exists() and RECORD_CSV_PATH.exists()):
    st.title("📡 The Booster Signal")
    st.error(
        f"Missing data files. Place them at:\n\n"
        f"- `{CLIENT_CSV_PATH}`\n- `{RECORD_CSV_PATH}`\n\n"
        f"or edit `CLIENT_CSV_PATH` / `RECORD_CSV_PATH` at the top of `app.py`."
    )
    st.markdown(f"**Expected columns** (merged on `Customer_ID`):\n\n`{', '.join(REQUIRED_COLS)}`")
    st.stop()

data_mtime_key = (CLIENT_CSV_PATH.stat().st_mtime, RECORD_CSV_PATH.stat().st_mtime)
df_raw = load_data(str(CLIENT_CSV_PATH), str(RECORD_CSV_PATH), data_mtime_key)
st.sidebar.success(f"Data loaded: {df_raw['Customer_ID'].nunique():,} subscribers.")

missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Data is missing required columns: {missing_cols}")
    st.stop()

df = engineer_features(df_raw)
FEATURES = [f for f in NUMERIC_FEATS_BASE if f in df.columns]

# ── Load model ─────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.title("📡 The Booster Signal")
    st.error(
        f"Missing model file. Place your Colab-trained model at:\n\n"
        f"- `{MODEL_PATH}`\n\n"
        f"(joblib/pickle `.pkl`/`.joblib`, or native XGBoost `.json`/`.ubj`), "
        f"or edit `MODEL_PATH` at the top of `app.py`."
    )
    st.stop()

model_mtime_key = MODEL_PATH.stat().st_mtime
try:
    pretrained_model, pretrained_features = load_pretrained_model(str(MODEL_PATH), model_mtime_key)
    st.sidebar.success(f"Model loaded: {MODEL_PATH.name}")
except Exception as e:
    st.error(f"Could not load model at `{MODEL_PATH}`: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Rupiah Financial Baseline")
ARPU_4G = st.sidebar.number_input("4G ARPU (Rp/mo)", 10_000, 200_000, 45_000, step=1_000)
ARPU_5G = st.sidebar.number_input("5G ARPU (Rp/mo)", 10_000, 300_000, 67_000, step=1_000)
CONTRACT_MONTHS = st.sidebar.slider("Contract horizon (months)", 3, 24, 12)
CAMPAIGN_CPM = st.sidebar.number_input("Cost per contact (Rp)", 50, 20_000, 500, step=50)
FP_PENALTY = st.sidebar.number_input("False-positive / opt-out penalty (Rp)", 0, 50_000, 2_000, step=100)

ARPU_UPLIFT = ARPU_5G - ARPU_4G
LTV_PER_CONVERT = ARPU_UPLIFT * CONTRACT_MONTHS

st.sidebar.markdown("---")
st.sidebar.markdown("### Scoring")
eval_fraction = st.sidebar.slider(
    "Evaluation holdout %", 0.05, 1.0, 0.20, step=0.05,
    help="Fraction of the data scored to populate the metrics, ROI, and decile-lift "
         "tabs. The model itself is never (re)trained here.",
)
seed = st.sidebar.number_input("Random seed (for the holdout split)", 0, 9999, 42)

data_fingerprint = (df.shape, float(df['Upsell_Target'].mean()), tuple(FEATURES))

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{NAVY}; padding: 22px 28px; border-radius:10px; margin-bottom:18px;">
  <h1 style="color:white; margin:0;">📡 The Booster Signal</h1>
  <p style="color:#B9CBE0; margin:4px 0 0 0; font-size:1.02rem;">
    Monetising the KOMDIGI Spectrum Auction Through ML-Driven 5G Upsell Targeting
    <span class="booster-badge">Interactive Analysis</span>
  </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Subscribers", f"{df['Customer_ID'].nunique():,}")
k2.metric("Prime 5G Candidates", f"{df['Upsell_Target'].sum():,}",
          f"{df['Upsell_Target'].mean()*100:.1f}% of base")
k3.metric("LTV / Convert", fmt_rp(LTV_PER_CONVERT))
k4.metric("Campaign Cost / Contact", fmt_rp(CAMPAIGN_CPM))

tabs = st.tabs([
    "🧭 Overview", "🔍 EDA", "🤖 Model & Optuna", "💰 Financial ROI",
    "🎯 Decile Lift", "📱 Go-to-Market", "⬇️ Export",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ──────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Strategic Framing")
    st.markdown("""
As of 2026, Telkomsel, Indosat Ooredoo Hutchison, and XL-Smartfren have poured
trillions of Rupiah into the KOMDIGI 700 MHz & 2.6 GHz spectrum auction, while ARPU
stays suppressed by price wars. The **"Paket Booster" overage signal** — subscribers
repeatedly buying emergency data top-ups — is the primary behavioural marker of
subscribers primed to convert to a premium 5G plan.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Column Translation: US Billing → Indonesian Behaviour")
        st.dataframe(pd.DataFrame({
            "Dataset column": ["ovrrev_Mean / ovrmou_Mean", "change_mou / change_rev",
                                "hnd_webcap", "eqpdays", "rev_Mean"],
            "US meaning": ["Overage revenue/minutes", "Change in usage/revenue",
                            "Handset web capability", "Equipment age (days)", "Mean revenue"],
            "Indonesian equivalent": [
                "Paket Booster spend/usage (emergency top-ups)",
                "Data hunger velocity / revenue momentum",
                "5G-ready handset (WCMB/WC)",
                "Time since last device upgrade",
                "Baseline ARPU tier",
            ],
        }), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("#### Financial Baseline")
        st.write(f"- Current 4G ARPU: **{fmt_rp(ARPU_4G)}**/mo")
        st.write(f"- Target 5G ARPU: **{fmt_rp(ARPU_5G)}**/mo")
        st.write(f"- Incremental uplift: **{fmt_rp(ARPU_UPLIFT)}**/mo")
        st.write(f"- Annual LTV / convert: **{fmt_rp(LTV_PER_CONVERT)}**")
        st.caption("Adjust these in the sidebar — every downstream tab recalculates live.")

    st.markdown("#### Target Definition")
    st.code(
        "Upsell_Target = (ovrrev_Mean > 0)              # buys Paket Booster\n"
        "               & hnd_webcap in ['WCMB','WC']    # 5G-capable handset\n"
        "               & (rev_Mean > market median)     # above baseline ARPU\n"
        "               & (churn == 0)                   # retained subscriber",
        language="python",
    )
    rev_median = df.attrs.get('rev_median', df['rev_Mean'].median())
    st.caption(f"Revenue threshold used: Rp {rev_median:,.0f} (market median of current data)")

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ──────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    # Plot 1: Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = df['Upsell_Target'].value_counts()
    labels = ['Standard Users (0)', 'Prime 5G Candidates (1)']
    axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)],
                color=[C_LIGHT, C_BLUE], edgecolor='white', linewidth=2, width=0.45)
    axes[0].set_title('Subscriber Classification')
    axes[0].set_ylabel('Subscribers')
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    axes[1].pie([counts.get(0, 0), counts.get(1, 0)], labels=labels,
                colors=[C_LIGHT, C_BLUE], autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'Class Balance (1 : {ratio:.1f})')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Paket Booster Pain Plot")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    grp_means = df.groupby('Upsell_Target')['ovrrev_Mean'].mean()
    axes[0].bar(['Standard', 'Prime 5G Candidates'], grp_means.values,
                color=[C_LIGHT, C_ORANGE], edgecolor='white', linewidth=2, width=0.45)
    axes[0].set_title('Avg Booster Spend by Group')
    axes[0].set_ylabel('Rp (proxy units)')
    sns.boxplot(x='Upsell_Target', y='booster_intensity', data=df,
                palette=[C_LIGHT, C_ORANGE], width=0.45, ax=axes[1])
    axes[1].set_title('Booster Intensity Distribution')
    axes[1].set_xticklabels(['Standard', 'Prime 5G'])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hardware Readiness")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        THRESHOLD_DAYS = 18 * 30
        sns.boxplot(x='Upsell_Target', y='eqpdays', data=df,
                    palette=[C_LIGHT, C_BLUE], width=0.45, ax=ax)
        ax.axhline(THRESHOLD_DAYS, color=C_RED, linestyle='--', label='18-month mark')
        ax.set_xticklabels(['Standard', 'Prime 5G'])
        ax.set_title('Equipment Age by Group')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("#### Data Hunger Velocity")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for grp, clr, lbl in [(0, C_LIGHT, 'Standard'), (1, C_ORANGE, 'Prime 5G')]:
            sub = df[df['Upsell_Target'] == grp]['data_hunger_velocity']
            ax.hist(sub, bins=40, alpha=0.6, color=clr, label=lbl, density=True)
        ax.set_xlabel('Δ Minutes of Use')
        ax.set_title('Data Hunger Velocity (change_mou)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Feature Correlation Heatmap")
    KEY_COLS = ['totmou', 'totrev', 'avgmou', 'avgrev', 'ovrmou_Mean', 'ovrrev_Mean',
                'eqpdays', 'booster_intensity', 'data_hunger_velocity',
                'rev_momentum', 'engagement_depth', 'booster_freq_proxy', 'Upsell_Target']
    avail = [c for c in KEY_COLS if c in df.columns]
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — Model & Optuna
# ──────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Pre-Trained Model — Scored, Not Retrained")
    st.caption(f"Loaded from `{MODEL_PATH.name}`. This tab scores it against a fresh "
               f"{eval_fraction*100:.0f}% slice of your Indonesian data — no fitting "
               f"happens here.")

    with st.spinner("Scoring model against current data…"):
        result = build_result_from_pretrained(
            pretrained_model, pretrained_features, df, eval_fraction, int(seed))
    st.session_state["model_result"] = result

    if result['missing_features']:
        st.warning("The model expects these columns, which aren't in your current "
                   f"data and were dropped: {result['missing_features']}")
    st.caption(f"Scored on **{len(result['features'])}** features: {', '.join(result['features'])}")

    model = result['model']
    X_test, y_test, y_prob = result['X_test'], result['y_test'], result['y_prob']
    y_pred_default = (y_prob >= 0.5).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    m2.metric("Precision (@0.5)", f"{precision_score(y_test, y_pred_default):.4f}")
    m3.metric("Recall (@0.5)", f"{recall_score(y_test, y_pred_default):.4f}")
    m4.metric("F1 (@0.5)", f"{f1_score(y_test, y_pred_default):.4f}")

    st.markdown("#### Feature Importance")
    if hasattr(model, "feature_importances_"):
        fi_df = (pd.DataFrame({'Feature': result['features'],
                                'Importance': model.feature_importances_})
                 .sort_values('Importance', ascending=True))
        colors = [C_ORANGE if f in BOOSTER_FEATS else C_LIGHT for f in fi_df['Feature']]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white')
        ax.set_title('⭐ Orange = Booster Signal Feature')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("This model type doesn't expose feature importances.")

    st.markdown("#### Confusion Matrix (@ 0.5 threshold)")
    cm = confusion_matrix(y_test, y_pred_default)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['Standard (0)', 'Prime 5G (1)']).plot(ax=ax, colorbar=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Full classification report"):
        st.text(classification_report(y_test, y_pred_default,
                                       target_names=['Standard', 'Prime 5G']))

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — Financial ROI
# ──────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Financial Cost Matrix — Live Rupiah ROI")
    if "model_result" not in st.session_state:
        st.warning("Train the model in the **Model & Optuna** tab first.")
        st.stop()

    result = st.session_state["model_result"]
    y_test, y_prob = result['y_test'], result['y_prob']

    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, step=0.01,
                           help="Move this to see precision/recall and ROI trade off in real time.")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    revenue_tp = tp * LTV_PER_CONVERT
    cost_fp = fp * (CAMPAIGN_CPM + FP_PENALTY)
    opp_fn = fn * LTV_PER_CONVERT * 0.50
    net_value = revenue_tp - cost_fp - opp_fn
    roi_pct = net_value / ((tp + fp) * CAMPAIGN_CPM) * 100 if (tp + fp) > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    m3.metric("Net Campaign Value", fmt_rp(net_value))
    m4.metric("Campaign ROI", f"{roi_pct:.1f}%")

    items = ['TP: Revenue\nCaptured', 'FP: Campaign\nWaste', 'FN: Missed\nRevenue (50%)', 'Net Campaign\nValue']
    values = [revenue_tp, -cost_fp, -opp_fn, net_value]
    bar_colors = [C_GREEN if v > 0 else C_RED for v in values]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(items, values, color=bar_colors, edgecolor='white', linewidth=2, width=0.5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_title(f'Financial Cost Matrix — Threshold {threshold:.2f} (Test Set)', fontweight='bold')
    ax.set_ylabel('Value (IDR)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x:,.0f}'))
    span = max(abs(v) for v in values) or 1
    for bar, val in zip(bars, values):
        offset = span * 0.03
        ypos = bar.get_height() + (offset if val >= 0 else -offset * 3)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'Rp {val:,.0f}',
                ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Full-Base ARPU Uplift Projection (CapEx Recovery Framing)")
    reach_pct = st.slider("Campaign reach — top X% by readiness score", 1, 50, 10)

    X_full = df[result['features']].fillna(df[result['features']].median())
    full_prob = model_full_prob = result['model'].predict_proba(X_full)[:, 1]
    df['readiness_score'] = full_prob

    thresh_val = np.percentile(full_prob, 100 - reach_pct)
    top_mask = full_prob >= thresh_val
    top_count = int(top_mask.sum())

    # decile-1-equivalent conversion rate estimated at current threshold slice of test set
    top_test_thresh = np.percentile(y_prob, 100 - reach_pct)
    test_top_mask = y_prob >= top_test_thresh
    top_conv_rate = y_test[test_top_mask].mean() if test_top_mask.sum() > 0 else y_test.mean()

    expected_converts = int(top_count * top_conv_rate)
    monthly_uplift = expected_converts * ARPU_UPLIFT
    annual_ltv = expected_converts * LTV_PER_CONVERT
    campaign_cost = top_count * CAMPAIGN_CPM
    net_roi = annual_ltv - campaign_cost
    roi_mult = annual_ltv / max(campaign_cost, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Top {reach_pct}% Reach", f"{top_count:,}")
    c2.metric("Expected Conversions", f"{expected_converts:,}", f"{top_conv_rate*100:.1f}% rate")
    c3.metric("Annual LTV Uplift", fmt_rp(annual_ltv))
    c4.metric("ROI Multiplier", f"{roi_mult:.1f}×")

    st.info(f"Every Rp 1 spent on this campaign returns **Rp {roi_mult:.1f}** in incremental "
            f"subscriber LTV — directly contributing to KOMDIGI spectrum CapEx recovery.")

# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — Decile Lift
# ──────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Decile Lift Analysis")
    if "model_result" not in st.session_state:
        st.warning("Train the model in the **Model & Optuna** tab first.")
        st.stop()

    result = st.session_state["model_result"]
    y_test, y_prob = result['y_test'], result['y_prob']

    n_bins = st.select_slider("Number of bins", options=[5, 10, 20], value=10)

    decile_df = (pd.DataFrame({'Actual': y_test.values, 'Prob': y_prob})
                 .sort_values('Prob', ascending=False).reset_index(drop=True))
    decile_df['Bin'] = pd.qcut(decile_df.index, n_bins, labels=range(1, n_bins + 1), duplicates='drop')

    summary = decile_df.groupby('Bin', observed=True).agg(
        n_subs=('Actual', 'count'), n_converts=('Actual', 'sum'), avg_prob=('Prob', 'mean')
    ).reset_index()
    summary['conv_rate'] = summary['n_converts'] / summary['n_subs']
    baseline_rate = y_test.mean()
    summary['lift'] = summary['conv_rate'] / baseline_rate if baseline_rate > 0 else 0

    top_quartile = max(1, n_bins // 4)
    c_dec = [C_ORANGE if i <= top_quartile else C_LIGHT for i in range(1, n_bins + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(summary['Bin'].astype(str), summary['conv_rate'] * 100, color=c_dec, edgecolor='white')
    axes[0].axhline(baseline_rate * 100, color=C_RED, linestyle='--', linewidth=2,
                     label=f'Mass-Blast Baseline ({baseline_rate*100:.1f}%)')
    axes[0].set_title('Conversion Rate by Predicted Bin')
    axes[0].set_xlabel('Bin (1 = Highest Probability)')
    axes[0].set_ylabel('Conversion Rate (%)')
    axes[0].legend()

    axes[1].bar(summary['Bin'].astype(str), summary['lift'], color=c_dec, edgecolor='white')
    axes[1].axhline(1.0, color=C_RED, linestyle='--', linewidth=2, label='No-Lift Baseline')
    axes[1].set_title('Lift Over Random Mass-Blast')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Lift (×)')
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_lift = summary.iloc[0]['lift']
    st.success(f"Top bin converts at **{top_lift:.2f}×** the mass-blast baseline rate.")
    st.dataframe(summary[['Bin', 'n_subs', 'n_converts', 'conv_rate', 'lift']].round(3),
                 hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 6 — Go-to-Market
# ──────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Indonesian E-Wallet Go-to-Market")
    if "model_result" not in st.session_state or 'readiness_score' not in df.columns:
        st.warning("Train the model in the **Model & Optuna** tab and visit the Financial ROI tab "
                   "first so readiness scores are computed.")
        st.stop()

    st.caption("Adjust tier thresholds to see how campaign reach & channel mix shift.")
    t1, t2, t3 = st.columns(3)
    tier1_cut = t1.slider("Tier 1 cutoff (GoPay/ShopeePay Intercept)", 0.5, 0.99, 0.80, step=0.01)
    tier2_cut = t2.slider("Tier 2 cutoff (MyTelkomsel Push)", 0.3, tier1_cut - 0.01, 0.60, step=0.01)
    tier3_cut = t3.slider("Tier 3 cutoff (In-App Banner)", 0.1, tier2_cut - 0.01, 0.40, step=0.01)

    def campaign_tier(score):
        if score >= tier1_cut: return 'Tier 1 — GoPay/ShopeePay Intercept'
        elif score >= tier2_cut: return 'Tier 2 — MyTelkomsel Push'
        elif score >= tier3_cut: return 'Tier 3 — In-App Banner'
        return 'Hold-out'

    def localised_message(row):
        tier = row['campaign_tier']
        n_booster = max(1, int(row['ovrmou_Mean'] / 60))
        if 'Tier 1' in tier:
            return (f"Kamu kehabisan kuota ±{n_booster}× bulan ini. "
                    f"Upgrade 5G Unlimited sekarang — hemat Rp 50.000 cashback via ShopeePay!")
        elif 'Tier 2' in tier:
            return "Paket Booster kamu makin sering. Coba 5G Premium hari ini—lebih murah per GB!"
        elif 'Tier 3' in tier:
            return "Nikmati kecepatan 5G. Lihat paket bundling Home+Mobile FMC kami."
        return None

    def ewallet_channel(tier):
        if 'Tier 1' in tier: return 'GoPay / ShopeePay API'
        if 'Tier 2' in tier: return 'MyTelkomsel / myIM3 Push'
        if 'Tier 3' in tier: return 'In-App Banner'
        return 'None'

    campaign_df = df[['Customer_ID', 'readiness_score', 'ovrrev_Mean',
                       'ovrmou_Mean', 'eqpdays', 'churn']].copy()
    campaign_df['campaign_tier'] = campaign_df['readiness_score'].apply(campaign_tier)
    campaign_df['message_id'] = campaign_df.apply(localised_message, axis=1)
    campaign_df['ewallet_channel'] = campaign_df['campaign_tier'].apply(ewallet_channel)

    tier_summary = (campaign_df.groupby('campaign_tier')
                     .agg(subscribers=('Customer_ID', 'count'),
                          avg_score=('readiness_score', 'mean'),
                          avg_booster_rev=('ovrrev_Mean', 'mean'))
                     .reset_index())
    tier_order = ['Tier 1 — GoPay/ShopeePay Intercept', 'Tier 2 — MyTelkomsel Push',
                  'Tier 3 — In-App Banner', 'Hold-out']
    tier_summary['campaign_tier'] = pd.Categorical(tier_summary['campaign_tier'],
                                                     categories=tier_order, ordered=True)
    tier_summary = tier_summary.sort_values('campaign_tier')

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### Campaign Tier Breakdown")
        st.dataframe(tier_summary.round(3), hide_index=True, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        tier_colors = [C_RED, C_ORANGE, C_LIGHT, "#CFD8DC"]
        ax.pie(tier_summary['subscribers'], labels=tier_summary['campaign_tier'],
               autopct='%1.0f%%', colors=tier_colors[:len(tier_summary)], startangle=90,
               textprops={'fontsize': 8})
        ax.set_title('Subscriber Mix by Tier')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Sample Localised Messages")
    sample = campaign_df[campaign_df['campaign_tier'] != 'Hold-out'].sort_values(
        'readiness_score', ascending=False).head(6)
    st.dataframe(sample[['Customer_ID', 'readiness_score', 'campaign_tier',
                          'ewallet_channel', 'message_id']].round(3),
                 hide_index=True, use_container_width=True)

    st.markdown("#### A/B Testing Simulation — E-Wallet Intercept vs Generic SMS")
    result = st.session_state["model_result"]
    y_test = result['y_test']
    baseline_rate = y_test.mean()
    decile_df_ab = (pd.DataFrame({'Actual': y_test.values, 'Prob': result['y_prob']})
                     .sort_values('Prob', ascending=False).reset_index(drop=True))
    decile_df_ab['Decile'] = pd.qcut(decile_df_ab.index, 10, labels=range(1, 11), duplicates='drop')
    top_conv_rate = decile_df_ab[decile_df_ab['Decile'] == 1]['Actual'].mean()

    rng = np.random.default_rng(int(seed))
    prime = campaign_df[campaign_df['campaign_tier'] == 'Tier 1 — GoPay/ShopeePay Intercept'].copy()
    if len(prime) == 0:
        st.warning("No subscribers fall in Tier 1 at the current cutoffs — lower the Tier 1 slider.")
    else:
        prime['ab_group'] = np.where(rng.random(len(prime)) < 0.5,
                                      'Treatment (E-Wallet Intercept)', 'Control (Generic SMS)')
        mask_t = prime['ab_group'] == 'Treatment (E-Wallet Intercept)'
        mask_c = ~mask_t
        prime['converted'] = 0
        prime.loc[mask_t, 'converted'] = rng.binomial(1, p=min(top_conv_rate * 1.8, 1), size=mask_t.sum())
        prime.loc[mask_c, 'converted'] = rng.binomial(1, p=baseline_rate, size=mask_c.sum())

        ab_res = prime.groupby('ab_group').agg(
            n=('Customer_ID', 'count'), conversions=('converted', 'sum'),
            conversion_rate=('converted', 'mean')
        ).reset_index()
        ab_res['revenue_idr'] = ab_res['conversions'] * LTV_PER_CONVERT
        st.dataframe(ab_res.round(4), hide_index=True, use_container_width=True)

        t = ab_res[ab_res['ab_group'] == 'Treatment (E-Wallet Intercept)']['conversion_rate'].values[0]
        c = ab_res[ab_res['ab_group'] == 'Control (Generic SMS)']['conversion_rate'].values[0]
        lift_pp = (t - c) * 100
        lift_mult = t / max(c, 0.0001)
        decision = "SCALE NATIONALLY via e-wallet APIs" if lift_pp >= 2 else "Continue pilot"
        st.write(f"**Incremental lift:** {lift_pp:.2f} pp  ·  **Relative lift:** {lift_mult:.2f}×  ·  **Decision:** {decision}")

    st.session_state["campaign_df"] = campaign_df

# ──────────────────────────────────────────────────────────────────────────
# TAB 7 — Export
# ──────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Export Campaign Targeting List")
    if "campaign_df" not in st.session_state:
        st.warning("Visit the **Go-to-Market** tab first to generate the campaign list.")
        st.stop()

    campaign_df = st.session_state["campaign_df"]
    export = (campaign_df[campaign_df['campaign_tier'] != 'Hold-out']
              [['Customer_ID', 'readiness_score', 'campaign_tier',
                'ewallet_channel', 'message_id', 'ovrrev_Mean', 'eqpdays']]
              .sort_values('readiness_score', ascending=False)
              .reset_index(drop=True))

    st.write(f"**{len(export):,}** subscribers targeted across active tiers.")
    st.dataframe(export.head(20).round(3), hide_index=True, use_container_width=True)

    csv_bytes = export.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download 5G_Indonesia_Campaign_List.csv",
        data=csv_bytes,
        file_name="5G_Indonesia_Campaign_List.csv",
        mime="text/csv",
        type="primary",
    )

    if "model_result" in st.session_state:
        result = st.session_state["model_result"]
        st.markdown("#### Model Summary")
        st.json({
            "AUC": round(float(roc_auc_score(result['y_test'], result['y_prob'])), 4),
            "features": result['features']
        })

st.markdown("---")
st.caption("The Booster Signal · Indonesian 5G Upsell Targeting Engine · "
           "Built with XGBoost + Optuna + SMOTE on the Streamlit framework.")
