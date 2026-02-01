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
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ChurnCatcher Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PREMIUM CSS STYLING ====================
# This creates the Netflix-style scrolling experience with premium visual polish
st.markdown("""
<style>
    /* ========== GLOBAL THEME ========== */
    /* Dark, rich base with gradient overlay */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove default padding for full-width sections */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* ========== SECTION CONTAINERS ========== */
    /* Each section gets its own visual identity */
    .section-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Alternate section styling for visual rhythm */
    .section-container-alt {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(168, 85, 247, 0.2);
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.2);
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* ========== ANIMATIONS ========== */
    /* Subtle fade-in on scroll simulation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* ========== HERO SECTION ========== */
    .hero-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border-radius: 30px;
        margin-bottom: 3rem;
        border: 2px solid rgba(168, 85, 247, 0.3);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    /* ========== SECTION HEADERS ========== */
    /* Storytelling-style headers with dynamic styling */
    .section-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-align: left;
        animation: slideInLeft 0.6s ease-out;
    }
    
    .section-subheader {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 400;
        margin-bottom: 2rem;
        font-style: italic;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* ========== METRIC CARDS (Netflix-style rows) ========== */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        animation: fadeInUp 0.6s ease-out;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(168, 85, 247, 0.4);
        border-color: rgba(168, 85, 247, 0.5);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ========== INSIGHT CARDS ========== */
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-left-color: #764ba2;
        transform: translateX(8px);
    }
    
    .insight-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
    }
    
    /* ========== NAVIGATION PILLS ========== */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 0.8rem 2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-pill:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: transparent;
        transform: scale(1.05);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* ========== STREAMLIT COMPONENT OVERRIDES ========== */
    /* Style native Streamlit elements to match theme */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
    }
    
    div[data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Style buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5);
    }
    
    /* Style dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .section-header {
            font-size: 1.8rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
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

# ==================== REUSABLE UI COMPONENTS ====================
def create_metric_card(value, label):
    """Create a Netflix-style metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_section_header(title, subtitle):
    """Create storytelling-style section header"""
    return f"""
    <div class="section-header">{title}</div>
    <div class="section-subheader">{subtitle}</div>
    """

def create_insight_card(title, text):
    """Create an insight card with hover effect"""
    return f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """

# ==================== MAIN APP ====================
def main():
    # ========== HERO SECTION ==========
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎯 ChurnCatcher</div>
        <div class="hero-subtitle">Predict. Prevent. Prosper.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = generate_sample_data(5000)
    
    # ========== SECTION 1: THE BIG PICTURE ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown(create_section_header(
        "Let's talk about the customers we're losing.",
        "Understanding churn is the first step to prevention."
    ), unsafe_allow_html=True)
    
    # Metric cards in horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card(f"{len(df):,}", "Total Customers"), unsafe_allow_html=True)
    with col2:
        churn_rate = df['churn'].mean() * 100
        st.markdown(create_metric_card(f"{churn_rate:.1f}%", "Churn Rate"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(f"{(df['churn'] == 0).sum():,}", "Retained"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(f"{(df['churn'] == 1).sum():,}", "At Risk"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== SECTION 2: VISUAL INSIGHTS ==========
    st.markdown('<div class="section-container-alt">', unsafe_allow_html=True)
    st.markdown(create_section_header(
        "Who's leaving, and why does it matter?",
        "Let the data tell the story."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df['churn'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=churn_counts.values,
            hole=0.5,
            marker_colors=['#667eea', '#f093fb'],
            textfont=dict(size=16, color='white')
        )])
        fig.update_layout(
            title=dict(text="Customer Distribution", font=dict(size=20, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index') * 100
        fig = px.bar(
            contract_churn,
            barmode='group',
            title="Churn by Contract Type",
            labels={'value': 'Percentage (%)', 'contract_type': 'Contract'},
            color_discrete_sequence=['#667eea', '#f093fb']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(size=20)),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== SECTION 3: KEY INSIGHTS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown(create_section_header(
        "What the numbers are telling us.",
        "Actionable insights from the data."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(create_insight_card(
            "💡 Month-to-Month Contracts",
            f"Customers on month-to-month plans are {((df[df['contract_type']=='Month-to-month']['churn'].mean() / df['churn'].mean() - 1) * 100):.0f}% more likely to churn."
        ), unsafe_allow_html=True)
        
        st.markdown(create_insight_card(
            "📞 Service Calls Matter",
            f"Customers with 5+ service calls have a {(df[df['customer_service_calls']>4]['churn'].mean()*100):.0f}% churn rate."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_insight_card(
            "⏰ First Year is Critical",
            f"{(df[df['tenure_months']<12]['churn'].mean()*100):.0f}% of customers churn within their first year."
        ), unsafe_allow_html=True)
        
        st.markdown(create_insight_card(
            "💰 Price Sensitivity",
            f"High-paying customers (>${df['monthly_charges'].quantile(0.75):.0f}/mo) show {(df[df['monthly_charges']>df['monthly_charges'].quantile(0.75)]['churn'].mean()*100):.0f}% churn rate."
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== SECTION 4: INTERACTIVE ANALYSIS ==========
    st.markdown('<div class="section-container-alt">', unsafe_allow_html=True)
    st.markdown(create_section_header(
        "Dive deeper into the patterns.",
        "Explore the relationships between customer behavior and churn."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x='churn', y='monthly_charges',
            title="Monthly Charges Impact",
            labels={'churn': 'Customer Status', 'monthly_charges': 'Monthly Charges ($)'},
            color='churn',
            color_discrete_sequence=['#667eea', '#f093fb']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(size=20)),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df, x='tenure_months', color='churn',
            title="Tenure Distribution",
            labels={'tenure_months': 'Months with Company', 'churn': 'Status'},
            color_discrete_sequence=['#667eea', '#f093fb'],
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(size=20)),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== SECTION 5: CALL TO ACTION ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown(create_section_header(
        "Ready to predict and prevent churn?",
        "Train ML models and start making intelligent predictions."
    ), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Train AI Models", use_container_width=True):
            with st.spinner("Training models... Creating your churn prediction engine..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                X, y, _ = preprocess_data(df)
                results, X_test, y_test, scaler = train_models(X, y)
                st.session_state['models'] = results
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                
                st.success("✨ Models trained! Scroll down to see performance.")
    
    if 'models' in st.session_state:
        st.markdown("---")
        results = st.session_state['models']
        
        st.markdown(create_section_header(
            "Your AI models are ready.",
            "Here's how they performed."
        ), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        for i, (name, result) in enumerate(results.items()):
            with [col1, col2, col3][i]:
                st.markdown(create_metric_card(
                    f"{result['accuracy']:.1%}",
                    f"{name}<br>Accuracy"
                ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ROC Curves
        fig = go.Figure()
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(st.session_state['y_test'], result['probabilities'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={result['auc']:.3f})",
                mode='lines',
                line=dict(width=3)
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Guess',
            mode='lines',
            line=dict(dash='dash', color='gray', width=2)
        ))
        fig.update_layout(
            title=dict(text="Model Performance Comparison", font=dict(size=24, color='white')),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()