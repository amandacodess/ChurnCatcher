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

# ==================== CONFIG ====================
st.set_page_config(
    page_title="ChurnCatcher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DESIGN SYSTEM ====================
COLORS = {
    'primary': '#2563eb',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'bg_light': '#f8fafc',
    'bg_card': '#ffffff',
    'text_primary': '#0f172a',
    'text_secondary': '#64748b',
    'border': '#e2e8f0'
}

# ==================== CUSTOM STYLING ====================
st.markdown(f"""
    <style>
    /* Global Resets */
    .main {{
        background-color: {COLORS['bg_light']};
    }}
    
    /* Remove default Streamlit padding */
    .block-container {{
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}
    
    /* Typography System */
    .hero-title {{
        font-size: 2rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    
    .hero-subtitle {{
        font-size: 1.125rem;
        font-weight: 400;
        color: {COLORS['text_secondary']};
        margin-bottom: 2.5rem;
    }}
    
    .section-title {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }}
    
    /* Metric Cards - Clean & Minimal */
    .metric-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        line-height: 1;
    }}
    
    .metric-change {{
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}
    
    .metric-change.positive {{
        color: {COLORS['success']};
    }}
    
    .metric-change.negative {{
        color: {COLORS['danger']};
    }}
    
    /* Chart Container */
    .chart-container {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-1px);
    }}
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Remove Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Divider */
    hr {{
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid {COLORS['border']};
    }}
    
    /* Info/Warning Boxes */
    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    /* Radio Buttons */
    .stRadio > label {{
        font-weight: 500;
        color: {COLORS['text_primary']};
    }}
    
    /* Dataframe Styling */
    .dataframe {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px;
    }}
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
        0.1 +
        0.3 * (df['contract_type'] == 'Month-to-month').astype(int) +
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
            'model': model,
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return results, X_test, y_test, scaler

def create_metric_card(label, value, change=None, change_type="neutral"):
    """Create a clean metric card with optional change indicator"""
    change_class = "positive" if change_type == "positive" else "negative" if change_type == "negative" else ""
    change_html = f'<div class="metric-change {change_class}">{change}</div>' if change else ''
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """

# ==================== PLOTLY THEME ====================
def get_plotly_layout(title):
    """Consistent Plotly layout configuration"""
    return {
        'title': {
            'text': title,
            'font': {'size': 16, 'weight': 600, 'color': COLORS['text_primary']},
            'x': 0,
            'xanchor': 'left'
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'system-ui, -apple-system, sans-serif', 'color': COLORS['text_primary']},
        'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20},
        'hoverlabel': {'bgcolor': 'white', 'font_size': 12},
        'xaxis': {'showgrid': False, 'showline': True, 'linecolor': COLORS['border']},
        'yaxis': {'showgrid': True, 'gridcolor': COLORS['border'], 'showline': False}
    }

# ==================== MAIN APP ====================
def main():
    # Hero Section
    st.markdown('<div class="hero-title">ChurnCatcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">ML-powered customer retention intelligence</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("", [
            "Overview",
            "Model Training",
            "Predictions",
            "Analytics"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### Data Source")
        data_source = st.radio("", ["Generate Sample Data", "Upload CSV"], label_visibility="collapsed")
        
        if data_source == "Generate Sample Data":
            n_samples = st.slider("Sample Size", 1000, 10000, 5000, 500)
            df = generate_sample_data(n_samples)
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.info("Upload a CSV file to continue")
                return
    
    # ==================== OVERVIEW PAGE ====================
    if page == "Overview":
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        churn_rate = df['churn'].mean() * 100
        retained = (df['churn'] == 0).sum()
        churned = (df['churn'] == 1).sum()
        
        with col1:
            st.markdown(create_metric_card("Total Customers", f"{total_customers:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Churn Rate", f"{churn_rate:.1f}%", 
                                          change_type="negative" if churn_rate > 20 else "positive"), 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Retained", f"{retained:,}"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("At Risk", f"{churned:,}"), unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Primary Insights</div>', unsafe_allow_html=True)
        
        # Primary Chart - Churn by Contract Type (most important insight)
        contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index') * 100
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Retained',
            x=contract_churn.index,
            y=contract_churn[0],
            marker_color=COLORS['success'],
            text=[f"{val:.1f}%" for val in contract_churn[0]],
            textposition='inside',
            textfont={'color': 'white', 'weight': 600}
        ))
        
        fig.add_trace(go.Bar(
            name='Churned',
            x=contract_churn.index,
            y=contract_churn[1],
            marker_color=COLORS['danger'],
            text=[f"{val:.1f}%" for val in contract_churn[1]],
            textposition='inside',
            textfont={'color': 'white', 'weight': 600}
        ))
        
        fig.update_layout(
            **get_plotly_layout("Retention Rate by Contract Type"),
            barmode='stack',
            showlegend=True,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
            height=400,
            yaxis_title="Percentage (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Supporting Charts
        st.markdown('<div class="section-title">Supporting Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tenure Distribution
            fig = go.Figure()
            
            for churn_val, color, name in [(0, COLORS['success'], 'Retained'), (1, COLORS['danger'], 'Churned')]:
                data = df[df['churn'] == churn_val]['tenure_months']
                fig.add_trace(go.Histogram(
                    x=data,
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig.update_layout(
                **get_plotly_layout("Tenure Distribution"),
                barmode='overlay',
                showlegend=True,
                xaxis_title="Tenure (months)",
                yaxis_title="Count",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly Charges Box Plot
            fig = go.Figure()
            
            for churn_val, color, name in [(0, COLORS['success'], 'Retained'), (1, COLORS['danger'], 'Churned')]:
                fig.add_trace(go.Box(
                    y=df[df['churn'] == churn_val]['monthly_charges'],
                    name=name,
                    marker_color=color,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                **get_plotly_layout("Monthly Charges Distribution"),
                showlegend=False,
                yaxis_title="Monthly Charges ($)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview (Collapsible)
        with st.expander("📋 View Raw Data", expanded=False):
            st.dataframe(df.head(50), use_container_width=True, height=400)
    
    # ==================== MODEL TRAINING PAGE ====================
    elif page == "Model Training":
        st.markdown('<div class="section-title">Model Training & Evaluation</div>', unsafe_allow_html=True)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                X, y, _ = preprocess_data(df)
                results, X_test, y_test, scaler = train_models(X, y)
                
                st.session_state['models'] = results
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                
                st.success("✓ Models trained successfully")
        
        if 'models' in st.session_state:
            results = st.session_state['models']
            
            # Model Performance Cards
            st.markdown("### Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            for i, (name, result) in enumerate(results.items()):
                with [col1, col2, col3][i]:
                    st.markdown(f"**{name}**")
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    st.metric("AUC-ROC", f"{result['auc']:.3f}")
            
            st.markdown("---")
            
            # ROC Curves
            st.markdown("### ROC Curves")
            fig = go.Figure()
            
            colors_map = {
                'Logistic Regression': COLORS['primary'],
                'Random Forest': COLORS['success'],
                'Gradient Boosting': COLORS['warning']
            }
            
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(st.session_state['y_test'], result['probabilities'])
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={result['auc']:.3f})",
                    mode='lines',
                    line={'width': 3, 'color': colors_map.get(name, COLORS['primary'])}
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Baseline',
                mode='lines',
                line={'dash': 'dash', 'color': COLORS['text_secondary'], 'width': 2}
            ))
            
            fig.update_layout(
                **get_plotly_layout("Model Comparison"),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            model_choice = st.selectbox("Select Model", list(results.keys()))
            
            cm = confusion_matrix(st.session_state['y_test'], results[model_choice]['predictions'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Retained', 'Predicted Churned'],
                y=['Actual Retained', 'Actual Churned'],
                colorscale=[[0, '#f8fafc'], [1, COLORS['primary']]],
                text=cm,
                texttemplate='<b>%{text}</b>',
                textfont={"size": 18},
                showscale=False
            ))
            
            fig.update_layout(
                **get_plotly_layout(f"{model_choice} - Confusion Matrix"),
                height=400,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PREDICTIONS PAGE ====================
    elif page == "Predictions":
        st.markdown('<div class="section-title">Make Predictions</div>', unsafe_allow_html=True)
        
        if 'models' not in st.session_state:
            st.warning("⚠ Please train models first")
            return
        
        pred_file = st.file_uploader("Upload customer data (CSV)", type=['csv'])
        
        if pred_file:
            pred_df = pd.read_csv(pred_file)
            
            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(pred_df.head(10), use_container_width=True)
            
            model_choice = st.selectbox("Select Model", list(st.session_state['models'].keys()))
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    X_pred, _, _ = preprocess_data(pred_df)
                    X_pred_scaled = st.session_state['scaler'].transform(X_pred)
                    
                    model = st.session_state['models'][model_choice]['model']
                    predictions = model.predict(X_pred_scaled)
                    probabilities = model.predict_proba(X_pred_scaled)[:, 1]
                    
                    pred_df['Churn_Prediction'] = predictions
                    pred_df['Churn_Probability'] = probabilities
                    pred_df['Risk_Level'] = pd.cut(probabilities, 
                                                    bins=[0, 0.3, 0.7, 1.0],
                                                    labels=['Low', 'Medium', 'High'])
                    
                    st.success("✓ Predictions complete")
                    
                    # Risk Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(create_metric_card("High Risk", 
                                                      f"{(pred_df['Risk_Level'] == 'High').sum():,}",
                                                      change_type="negative"), 
                                   unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric_card("Medium Risk", 
                                                      f"{(pred_df['Risk_Level'] == 'Medium').sum():,}",
                                                      change_type="neutral"), 
                                   unsafe_allow_html=True)
                    with col3:
                        st.markdown(create_metric_card("Low Risk", 
                                                      f"{(pred_df['Risk_Level'] == 'Low').sum():,}",
                                                      change_type="positive"), 
                                   unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### Prediction Results")
                    st.dataframe(pred_df, use_container_width=True, height=400)
                    
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Upload a CSV file with customer data to generate predictions")
    
    # ==================== ANALYTICS PAGE ====================
    elif page == "Analytics":
        st.markdown('<div class="section-title">Advanced Analytics</div>', unsafe_allow_html=True)
        
        if 'models' not in st.session_state:
            st.warning("⚠ Please train models first")
            return
        
        # Feature Importance
        st.markdown("### Feature Importance")
        
        if 'Random Forest' in st.session_state['models']:
            model = st.session_state['models']['Random Forest']['model']
            X, y, _ = preprocess_data(df)
            
            importances = model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color=COLORS['primary'],
                text=[f"{val:.3f}" for val in importance_df['Importance']],
                textposition='outside'
            ))
            
            fig.update_layout(
                **get_plotly_layout("Top 10 Features (Random Forest)"),
                height=450,
                xaxis_title="Importance Score",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Customer Segmentation
        st.markdown("### Risk Segmentation")
        
        X, y, _ = preprocess_data(df)
        X_scaled = st.session_state['scaler'].transform(X)
        model = st.session_state['models']['Random Forest']['model']
        
        df['Churn_Probability'] = model.predict_proba(X_scaled)[:, 1]
        df['Risk_Level'] = pd.cut(df['Churn_Probability'], 
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['Low', 'Medium', 'High'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = df['Risk_Level'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.5,
                marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']],
                textinfo='label+percent',
                textfont={'size': 14, 'weight': 600}
            )])
            
            fig.update_layout(
                **get_plotly_layout("Customer Distribution by Risk"),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            
            for risk, color in [('Low', COLORS['success']), 
                               ('Medium', COLORS['warning']), 
                               ('High', COLORS['danger'])]:
                risk_data = df[df['Risk_Level'] == risk]
                fig.add_trace(go.Scatter(
                    x=risk_data['tenure_months'],
                    y=risk_data['monthly_charges'],
                    mode='markers',
                    name=risk,
                    marker={'color': color, 'size': 6, 'opacity': 0.6}
                ))
            
            fig.update_layout(
                **get_plotly_layout("Risk by Tenure & Charges"),
                xaxis_title="Tenure (months)",
                yaxis_title="Monthly Charges ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()