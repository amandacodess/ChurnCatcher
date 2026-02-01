import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="ChurnCatcher",
    page_icon="🎯",
    layout="wide"
)

# ===================== THEME TOGGLE =====================
if "dark" not in st.session_state:
    st.session_state.dark = False

st.sidebar.title("Settings")
st.session_state.dark = st.sidebar.toggle("🌙 Dark mode", value=st.session_state.dark)

# ===================== COLORS =====================
if st.session_state.dark:
    BG = "#0b1220"
    CARD = "#111827"
    TEXT = "#e5e7eb"
    MUTED = "#9ca3af"
    ACCENT = "#3b82f6"
else:
    BG = "#f8fafc"
    CARD = "#ffffff"
    TEXT = "#0f172a"
    MUTED = "#64748b"
    ACCENT = "#2563eb"

# ===================== CSS =====================
st.markdown(f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.block-container {{
    max-width: 1200px;
}}

.card {{
    background: {CARD};
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}}

.hero {{
    text-align: center;
    margin: 3rem 0 4rem 0;
}}

.hero h1 {{
    font-size: 3rem;
    font-weight: 800;
}}

.hero p {{
    font-size: 1.2rem;
    color: {MUTED};
}}

.kpi-label {{
    font-size: 0.75rem;
    color: {MUTED};
    text-transform: uppercase;
}}

.kpi-value {{
    font-size: 2rem;
    font-weight: 700;
}}

section[data-testid="stSidebar"] {{
    background-color: {CARD};
}}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
@st.cache_data
def load_data(n=5000):
    np.random.seed(42)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 150, n),
        "service_calls": np.random.randint(0, 10, n),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
    })
    churn_prob = (
        0.30 * (df["contract"] == "Month-to-month") +
        0.25 * (df["tenure"] < 12) +
        0.20 * (df["service_calls"] > 4)
    )
    df["churn"] = (np.random.rand(n) < churn_prob).astype(int)
    return df

df = load_data()

# ===================== HERO =====================
st.markdown("""
<div class="hero">
    <h1>ChurnCatcher</h1>
    <p>Predict churn. Understand behavior. Reduce revenue loss.</p>
</div>
""", unsafe_allow_html=True)

# ===================== PROBLEM → INSIGHT → IMPACT =====================
st.markdown("## The Business Case")
c1, c2, c3 = st.columns(3)

for col, title, text in zip(
    [c1, c2, c3],
    ["Problem", "Insight", "Impact"],
    [
        "Churn is expensive and often detected too late.",
        "Usage behavior predicts churn early.",
        "Targeted retention increases lifetime value."
    ]
):
    col.markdown(f"""
    <div class="card">
        <h3>{title}</h3>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

# ===================== KPIs =====================
st.markdown("## Executive Overview")
k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"<div class='card'><div class='kpi-label'>Customers</div><div class='kpi-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'><div class='kpi-label'>Churn Rate</div><div class='kpi-value'>{df['churn'].mean()*100:.1f}%</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'><div class='kpi-label'>High Risk</div><div class='kpi-value'>{(df['tenure']<12).sum():,}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'><div class='kpi-label'>Avg Charges</div><div class='kpi-value'>${df['monthly_charges'].mean():.0f}</div></div>", unsafe_allow_html=True)

# ===================== ANALYTICS =====================
st.markdown("## Behavioral Insights")
a1, a2 = st.columns(2)

fig = px.box(df, x="churn", y="monthly_charges", color="churn")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
a1.plotly_chart(fig, use_container_width=True)

fig = px.histogram(df, x="tenure", color="churn", opacity=0.7)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
a2.plotly_chart(fig, use_container_width=True)

# ===================== MODELING =====================
st.markdown("## Model Performance")

df_model = df.copy()
df_model["contract"] = LabelEncoder().fit_transform(df_model["contract"])

X = df_model.drop("churn", axis=1)
y = df_model["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150),
    "Gradient Boosting": GradientBoostingClassifier()
}

roc_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    roc_results[name] = (probs, roc_auc_score(y_test, probs))

# ===================== ROC CURVE =====================
fig = go.Figure()
for name, (probs, auc) in roc_results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.2f})"))

fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Baseline"))
fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT)
)
st.plotly_chart(fig, use_container_width=True)

# ===================== PREDICTION PANEL =====================
st.markdown("## Predict a Customer")

with st.form("predict"):
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    charges = st.slider("Monthly Charges", 20, 150, 70)
    calls = st.slider("Service Calls", 0, 10, 2)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    submit = st.form_submit_button("Predict")

if submit:
    contract_enc = LabelEncoder().fit(df["contract"]).transform([contract])[0]
    input_data = scaler.transform([[tenure, charges, calls, contract_enc]])
    prob = models["Gradient Boosting"].predict_proba(input_data)[0][1]

    st.markdown(f"""
    <div class="card">
        <h3>Churn Risk</h3>
        <h1>{prob*100:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)
