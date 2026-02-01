import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ChurnCatcher Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== REFINED DESIGN SYSTEM ====================
st.markdown("""
<style>
    /* ========== DESIGN TOKENS ========== */
    :root {
        --primary: #6366F1;        /* Single brand color - Indigo */
        --primary-hover: #4F46E5;
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
        
        --bg-primary: #FFFFFF;
        --bg-secondary: #F9FAFB;
        --bg-tertiary: #F3F4F6;
        
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --text-tertiary: #9CA3AF;
        
        --border: #E5E7EB;
        --border-hover: #D1D5DB;
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    
    /* ========== GLOBAL RESET ========== */
    .stApp {
        background: var(--bg-secondary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
    }
    
    /* ========== TYPOGRAPHY SYSTEM ========== */
    .hero-section {
        text-align: center;
        padding: 4rem 0 3rem;
        margin-bottom: 3rem;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .section-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* ========== CARD SYSTEM ========== */
    .metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-change {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        color: var(--success);
    }
    
    .metric-change.negative {
        color: var(--danger);
    }
    
    /* ========== INSIGHT CARDS ========== */
    .insight-card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    
    .insight-card:hover {
        border-color: var(--primary);
    }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .insight-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        font-size: 0.9375rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ========== SECTIONS ========== */
    .content-section {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .divider {
        height: 1px;
        background: var(--border);
        margin: 2rem 0;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background: var(--primary-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* ========== STREAMLIT OVERRIDES ========== */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.875rem;
    }
    
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* ========== CHART CONTAINER ========== */
    .chart-container {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    /* ========== UTILITIES ========== */
    .text-center {
        text-align: center;
    }
    
    .mb-1 { margin-bottom: 0.5rem; }
    .mb-2 { margin-bottom: 1rem; }
    .mb-3 { margin-bottom: 1.5rem; }
    .mb-4 { margin-bottom: 2rem; }
    
    .mt-1 { margin-top: 0.5rem; }
    .mt-2 { margin-top: 1rem; }
    .mt-3 { margin-top: 1.5rem; }
    .mt-4 { margin-top: 2rem; }
    
    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def generate_sample_data(n_samples=5000):
    """Generate synthetic customer churn data"""
    np.random.seed(42)
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No'], n_samples),
        'num_services': np.random.randint(0, 6, n_samples),
        'customer_service_calls': np.random.randint(0, 10, n_samples)
    }
    df = pd.DataFrame(data)
    churn_prob = (
        0.1 + 0.3 * (df['contract_type'] == 'Month-to-month').astype(int) +
        0.2 * (df['tenure_months'] < 12).astype(int) +
        0.15 * (df['customer_service_calls'] > 4).astype(int) +
        0.1 * (df['monthly_charges'] > 100).astype(int) -
        0.2 * (df['contract_type'] == 'Two year').astype(int)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    return df

def preprocess_data(df):
    """Preprocess data for modeling"""
    df_processed = df.drop('customer_id', axis=1, errors='ignore')
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    if 'churn' in df_processed.columns:
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        return X, y, label_encoders
    else:
        return df_processed, None, label_encoders

@st.cache_resource
def train_models(X, y):
    """Train multiple models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = model.score(X_test_scaled, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        results[name] = {
            'model': model, 'accuracy': accuracy, 'auc': auc_score,
            'predictions': y_pred, 'probabilities': y_pred_proba
        }
    return results, X_test, y_test, scaler

# ==================== UI COMPONENTS ====================
def create_metric_card(value, label, change=None):
    """Clean metric card with optional change indicator"""
    change_html = ""
    if change:
        change_class = "positive" if change > 0 else "negative"
        change_symbol = "↑" if change > 0 else "↓"
        change_html = f'<div class="metric-change {change_class}">{change_symbol} {abs(change):.1f}%</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """

def create_insight_card(icon, title, text):
    """Clean insight card"""
    return f"""
    <div class="insight-card">
        <div class="insight-icon">{icon}</div>
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """

# ==================== MAIN APP ====================
def main():
    # Load data
    df = generate_sample_data(5000)
    churn_rate = df['churn'].mean() * 100
    
    # ========== HERO SECTION ==========
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">ChurnCatcher Analytics</h1>
        <p class="hero-subtitle">Intelligent customer retention insights powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== KEY METRICS ==========
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Current customer retention metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card(f"{len(df):,}", "Total Customers"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(f"{churn_rate:.1f}%", "Churn Rate", change=-2.3), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"{(df['churn'] == 0).sum():,}", "Active Customers"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"{(df['churn'] == 1).sum():,}", "At Risk"), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== INSIGHTS SECTION ==========
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Data-driven observations from customer behavior</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        month_to_month_churn = df[df['contract_type']=='Month-to-month']['churn'].mean()
        increase = ((month_to_month_churn / df['churn'].mean() - 1) * 100)
        st.markdown(create_insight_card(
            "📋",
            "Contract Type Impact",
            f"Month-to-month customers show {increase:.0f}% higher churn rate compared to long-term contracts."
        ), unsafe_allow_html=True)
        
        high_service_calls_churn = df[df['customer_service_calls']>4]['churn'].mean() * 100
        st.markdown(create_insight_card(
            "📞",
            "Service Quality Indicator",
            f"Customers with 5+ support calls have {high_service_calls_churn:.0f}% churn probability."
        ), unsafe_allow_html=True)
    
    with col2:
        first_year_churn = df[df['tenure_months']<12]['churn'].mean() * 100
        st.markdown(create_insight_card(
            "⏱️",
            "Critical Onboarding Period",
            f"{first_year_churn:.0f}% of churns occur within the first 12 months of service."
        ), unsafe_allow_html=True)
        
        high_paying_churn = df[df['monthly_charges']>df['monthly_charges'].quantile(0.75)]['churn'].mean() * 100
        st.markdown(create_insight_card(
            "💰",
            "Price Sensitivity",
            f"High-value customers (top 25%) show {high_paying_churn:.0f}% churn rate despite premium pricing."
        ), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== ANALYTICS SECTION ==========
    st.markdown('<div class="section-title">Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Distribution and patterns in customer data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-title">Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df['churn'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Active', 'Churned'],
            values=churn_counts.values,
            hole=0.4,
            marker_colors=['#6366F1', '#EF4444'],
            textfont=dict(size=14)
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="chart-title">Churn by Contract Type</div>', unsafe_allow_html=True)
        contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index') * 100
        fig = px.bar(
            contract_churn,
            barmode='group',
            labels={'value': 'Percentage (%)', 'contract_type': 'Contract Type'},
            color_discrete_sequence=['#6366F1', '#EF4444']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=60),
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title="")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="mt-3"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-title">Monthly Charges Impact</div>', unsafe_allow_html=True)
        fig = px.box(
            df, x='churn', y='monthly_charges',
            labels={'churn': 'Customer Status', 'monthly_charges': 'Monthly Charges ($)'},
            color='churn',
            color_discrete_sequence=['#6366F1', '#EF4444']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=40),
            height=350,
            showlegend=False
        )
        fig.update_xaxes(ticktext=['Active', 'Churned'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="chart-title">Customer Tenure Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x='tenure_months', color='churn',
            labels={'tenure_months': 'Months with Company', 'churn': 'Status'},
            color_discrete_sequence=['#6366F1', '#EF4444'],
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=40),
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title="")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== MODEL TRAINING SECTION ==========
    st.markdown('<div class="section-title">Predictive Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Train machine learning models to predict customer churn</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Train Models", use_container_width=True, type="primary"):
            with st.spinner("Training models..."):
                X, y, _ = preprocess_data(df)
                results, X_test, y_test, scaler = train_models(X, y)
                st.session_state['models'] = results
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.success("✓ Models trained successfully")
    
    # Display results if models are trained
    if 'models' in st.session_state:
        st.markdown('<div class="mt-4"></div>', unsafe_allow_html=True)
        results = st.session_state['models']
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        for i, (name, result) in enumerate(results.items()):
            with [col1, col2, col3][i]:
                st.markdown(create_metric_card(
                    f"{result['accuracy']:.1%}",
                    name.upper()
                ), unsafe_allow_html=True)
        
        st.markdown('<div class="mt-3"></div>', unsafe_allow_html=True)
        
        # ROC Curve
        st.markdown('<div class="chart-title">Model Performance Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        
        colors = {'Logistic Regression': '#6366F1', 'Random Forest': '#10B981', 'Gradient Boosting': '#F59E0B'}
        
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(st.session_state['y_test'], result['probabilities'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={result['auc']:.3f})",
                mode='lines',
                line=dict(width=2, color=colors.get(name, '#6366F1'))
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='#9CA3AF', width=1)
        ))
        
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()