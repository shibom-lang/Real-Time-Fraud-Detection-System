"""
Streamlit Fraud Detection Dashboard — app.py
Multi-page fraud operations dashboard with live risk scoring and SHAP explainer.
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb

# ── Page configuration 
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model.pkl only (no preprocessed.pkl needed) ─────────────────────────
@st.cache_resource
def load_artifacts():
    """Load trained model and test data from model.pkl."""
    with open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb") as f:
        m = pickle.load(f)
    return m

@st.cache_data
def build_transactions_df(_m):
    """Build transactions DataFrame from the saved test set."""
    feature_cols = _m["feature_cols"]
    df = pd.DataFrame(_m["X_test"], columns=feature_cols)
    df["TransactionID"] = np.arange(len(df))
    df["isFraud"]       = _m["y_test"]
    df["fraud_prob"]    = _m["lgb_prob"]
    df["RiskTier"] = df["fraud_prob"].apply(
        lambda p: "🔴 Critical" if p >= 0.75 else ("🟡 Suspicious" if p >= 0.40 else "🟢 Clear")
    )
    return df

m      = load_artifacts()
df_all = build_transactions_df(m)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/fraud.png", width=60)
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.caption("IEEE-CIS Fraud Detection — LightGBM")

page = st.sidebar.radio(
    "Navigate",
    [" Overview", "🔎 Transaction Explorer", "🧠 SHAP Explainer"],
)

st.sidebar.markdown("---")
tier_filter = st.sidebar.multiselect(
    "Risk Tier Filter",
    options=["🔴 Critical", "🟡 Suspicious", "🟢 Clear"],
    default=["🔴 Critical", "🟡 Suspicious", "🟢 Clear"],
)
min_prob = st.sidebar.slider("Min Fraud Probability", 0.0, 1.0, 0.0, 0.01)

# Apply sidebar filters
df_view = df_all[
    (df_all["RiskTier"].isin(tier_filter)) &
    (df_all["fraud_prob"] >= min_prob)
]

# ── Page 1 — Overview ─────────────────────────────────────────────────────────
if page == " Overview":
    st.title(" Fraud Operations — Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df_all):,}")
    col2.metric("Fraud Count",        f"{df_all['isFraud'].sum():,}")
    col3.metric("Detection Rate",     f"{df_all['isFraud'].mean()*100:.2f}%")
    col4.metric("Avg Prob (Critical)",
                f"{df_all[df_all['RiskTier']=='🔴 Critical']['fraud_prob'].mean():.3f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Risk tier donut
    tier_counts = df_all["RiskTier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig_donut = px.pie(
        tier_counts, names="Tier", values="Count", hole=0.45,
        title="Risk Tier Distribution",
        color="Tier",
        color_discrete_map={"🔴 Critical": "#F44336",
                            "🟡 Suspicious": "#FFC107",
                            "🟢 Clear": "#4CAF50"}
    )
    fig_donut.update_traces(textposition="outside", textinfo="percent+label")
    c1.plotly_chart(fig_donut, use_container_width=True)

    # Fraud rate by hour
    if "HourOfDay" in df_all.columns:
        hourly = df_all.groupby("HourOfDay")["isFraud"].mean().reset_index()
        hourly.columns = ["Hour", "FraudRate"]
        fig_hour = px.bar(
            hourly, x="Hour", y="FraudRate",
            title="Fraud Rate by Hour of Day",
            color="FraudRate", color_continuous_scale="RdYlGn_r",
            labels={"FraudRate": "Fraud Rate", "Hour": "Hour of Day"}
        )
        c2.plotly_chart(fig_hour, use_container_width=True)

    # Fraud probability histogram
    fig_hist = px.histogram(
        df_all, x="fraud_prob", nbins=80, color="RiskTier",
        color_discrete_map={"🔴 Critical": "#F44336",
                            "🟡 Suspicious": "#FFC107",
                            "🟢 Clear": "#4CAF50"},
        title="Distribution of Fraud Probability Scores",
        labels={"fraud_prob": "Fraud Probability"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Page 2 — Transaction Explorer ────────────────────────────────────────────
elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")

    search_id = st.text_input("Search by TransactionID", placeholder="e.g. 42")

    if search_id.strip():
        try:
            tid    = int(search_id.strip())
            result = df_all[df_all["TransactionID"] == tid]
            if not result.empty:
                row = result.iloc[0]
                st.success(f"TransactionID {tid} found")
                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{row['fraud_prob']:.4f}")
                c2.metric("Risk Tier",         row["RiskTier"])
                c3.metric("Actual Label", "FRAUD" if row["isFraud"] == 1 else "LEGITIMATE")
            else:
                st.error(f"TransactionID {tid} not found.")
        except ValueError:
            st.warning("Please enter a valid integer TransactionID.")

    st.markdown("---")
    st.write(f"Showing **{len(df_view):,}** transactions (filters applied)")

    display_cols = ["TransactionID", "fraud_prob", "RiskTier", "isFraud",
                    "HourOfDay", "AmtToMeanRatio"]
    display_cols = [c for c in display_cols if c in df_view.columns]

    st.dataframe(
        df_view[display_cols]
            .sort_values("fraud_prob", ascending=False)
            .head(500)
            .style.background_gradient(subset=["fraud_prob"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=450,
    )

    csv = df_view[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download Filtered Results", csv,
                       "filtered_transactions.csv", "text/csv")

# ── Page 3 — SHAP Explainer ───────────────────────────────────────────────────
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer — Live Predictions")

    txn_id = st.number_input(
        "Enter TransactionID for explanation",
        min_value=0,
        max_value=int(df_all["TransactionID"].max()),
        value=0, step=1,
    )

    if st.button("🔍 Explain this Transaction"):
        row = df_all[df_all["TransactionID"] == txn_id]
        if row.empty:
            st.error("TransactionID not found.")
        else:
            prob   = row.iloc[0]["fraud_prob"]
            tier   = row.iloc[0]["RiskTier"]
            actual = "FRAUD" if row.iloc[0]["isFraud"] else "LEGITIMATE"

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{prob:.4f}")
            c2.metric("Risk Tier",         tier)
            c3.metric("Actual",            actual)

            feature_cols = m["feature_cols"]
            X_row        = row[feature_cols].values.astype(np.float32)

            with st.spinner("Computing SHAP values …"):
                explainer   = shap.TreeExplainer(m["lgb_model"])
                shap_values = explainer(X_row)
                exp = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.title(f"SHAP Waterfall — TransactionID {txn_id} (prob={prob:.4f})",
                      fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

            # Plain-English explanation
            st.markdown("### 📝 Plain-English Explanation")
            sv_arr  = exp.values
            fn_arr  = np.array(feature_cols)
            top_pos = fn_arr[np.argsort(sv_arr)[::-1][:3]]
            top_neg = fn_arr[np.argsort(sv_arr)[:3]]

            if prob >= 0.75:
                st.error(
                    f"🔴 **CRITICAL RISK** — {prob*100:.1f}% confident this is fraud. "
                    f"Top fraud drivers: **{', '.join(top_pos)}**."
                )
            elif prob >= 0.40:
                st.warning(
                    f"🟡 **SUSPICIOUS** — Mixed signals. "
                    f"Raising risk: **{', '.join(top_pos)}**. "
                    f"Reducing risk: **{', '.join(top_neg)}**. Manual review recommended."
                )
            else:
                st.success(
                    f"🟢 **CLEAR** — {(1-prob)*100:.1f}% confident this is legitimate. "
                    f"Key features suppressing fraud: **{', '.join(top_neg)}**."
                )