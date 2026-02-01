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
import io
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ChurnCatcher 🎯",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ===== FULL PAGE GRADIENT BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        120deg,
        #0f0c29,
        #302b63,
        #24243e
    );
    background-size: 300% 300%;
    animation: gradientShift 18s ease infinite;
}

/* Subtle animated movement */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Keep content centered and breathable */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

/* ===== HEADERS ===== */
.main-header {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #7f7cff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.sub-header {
    text-align: center;
    color: #d1d5db;
    margin-bottom: 3rem;
}

/* ===== GLASS CARDS ===== */
.metric-card,
div[data-testid="stMetric"],
div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"],
div[data-testid="stContainer"] > div {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* ===== METRICS ===== */
div[data-testid="stMetric"] {
    padding: 1.2rem;
    text-align: center;
}

div[data-testid="stMetric"] label {
    color: #c7d2fe !important;
}

div[data-testid="stMetric"] div {
    color: #ffffff !important;
    font-weight: 700;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(30,27,75,0.95),
        rgba(15,23,42,0.95)
    );
    border-right: 1px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(168,85,247,0.4);
}

/* ===== DATAFRAME ===== */
.dataframe {
    border-radius: 14px;
    overflow: hidden;
}

/* ===== REMOVE STREAMLIT WHITES ===== */
header, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# Helper Functions
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
    
    # Encode categorical variables
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

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🎯 ChurnCatcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Customer Churn with Machine Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    page = st.sidebar.radio("Navigation", [
        "📊 Dashboard",
        "🤖 Model Training",
        "🔮 Predictions",
        "📈 Analytics"
    ])
    
    # Load or generate data
    data_source = st.sidebar.radio("Data Source", ["Generate Sample Data", "Upload CSV"])
    
    if data_source == "Generate Sample Data":
        n_samples = st.sidebar.slider("Number of Samples", 1000, 10000, 5000, 500)
        df = generate_sample_data(n_samples)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or select 'Generate Sample Data'")
            return
    
    # Dashboard Page
    if page == "📊 Dashboard":
        st.header("📊 Customer Churn Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churn_rate = df['churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Retained", f"{(df['churn'] == 0).sum():,}")
        with col4:
            st.metric("Churned", f"{(df['churn'] == 1).sum():,}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn Distribution
            churn_counts = df['churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Retained', 'Churned'],
                values=churn_counts.values,
                hole=0.4,
                marker_colors=['#2ecc71', '#e74c3c']
            )])
            fig.update_layout(title="Churn Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn by Contract Type
            contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index') * 100
            fig = px.bar(
                contract_churn,
                barmode='group',
                title="Churn Rate by Contract Type",
                labels={'value': 'Percentage (%)', 'contract_type': 'Contract Type'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Charges Distribution
            fig = px.box(df, x='churn', y='monthly_charges', 
                        title="Monthly Charges by Churn Status",
                        labels={'churn': 'Churn (0=No, 1=Yes)', 'monthly_charges': 'Monthly Charges ($)'},
                        color='churn',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tenure Distribution
            fig = px.histogram(df, x='tenure_months', color='churn',
                             title="Tenure Distribution by Churn",
                             labels={'tenure_months': 'Tenure (months)', 'churn': 'Churned'},
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                             barmode='overlay')
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
    
    # Model Training Page
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment..."):
                X, y, _ = preprocess_data(df)
                results, X_test, y_test, scaler = train_models(X, y)
                
                # Store in session state
                st.session_state['models'] = results
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                
                st.success("✅ Models trained successfully!")
        
        if 'models' in st.session_state:
            results = st.session_state['models']
            
            st.subheader("📊 Model Performance Comparison")
            
            # Performance Metrics
            col1, col2, col3 = st.columns(3)
            
            for i, (name, result) in enumerate(results.items()):
                with [col1, col2, col3][i]:
                    st.markdown(f"### {name}")
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    st.metric("AUC-ROC", f"{result['auc']:.3f}")
            
            st.markdown("---")
            
            # ROC Curves
            st.subheader("📈 ROC Curves")
            fig = go.Figure()
            
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(st.session_state['y_test'], result['probabilities'])
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={result['auc']:.3f})",
                    mode='lines'
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title="ROC Curves - Model Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            model_choice = st.selectbox("Select Model", list(results.keys()))
            
            cm = confusion_matrix(st.session_state['y_test'], results[model_choice]['predictions'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Retained', 'Churned'],
                y=['Retained', 'Churned'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            fig.update_layout(
                title=f"Confusion Matrix - {model_choice}",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Predictions Page
    elif page == "🔮 Predictions":
        st.header("🔮 Make Predictions")
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'Model Training' page!")
            return
        
        st.subheader("Upload Customer Data for Prediction")
        
        pred_file = st.file_uploader("Upload CSV with customer data", type=['csv'], key='prediction')
        
        if pred_file:
            pred_df = pd.read_csv(pred_file)
            st.write("📋 Uploaded Data Preview:")
            st.dataframe(pred_df.head(), use_container_width=True)
            
            model_choice = st.selectbox("Select Model for Prediction", list(st.session_state['models'].keys()))
            
            if st.button("🎯 Predict Churn", type="primary"):
                with st.spinner("Making predictions..."):
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
                    
                    st.success("✅ Predictions completed!")
                    
                    # Results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk Customers", (pred_df['Risk_Level'] == 'High').sum())
                    with col2:
                        st.metric("Medium Risk Customers", (pred_df['Risk_Level'] == 'Medium').sum())
                    with col3:
                        st.metric("Low Risk Customers", (pred_df['Risk_Level'] == 'Low').sum())
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download button
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("💡 Tip: Upload a CSV file with customer data (without the 'churn' column) to make predictions!")
    
    # Analytics Page
    elif page == "📈 Analytics":
        st.header("📈 Advanced Analytics")
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'Model Training' page!")
            return
        
        # Feature Importance
        st.subheader("🎯 Feature Importance Analysis")
        
        # Get feature importance from Random Forest
        if 'Random Forest' in st.session_state['models']:
            model = st.session_state['models']['Random Forest']['model']
            X, y, _ = preprocess_data(df)
            
            importances = model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h',
                        title="Feature Importance (Random Forest)",
                        color='Importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Customer Segmentation
        st.subheader("👥 Customer Segmentation by Risk")
        
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
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title="Customer Distribution by Risk Level",
                        color=risk_counts.index,
                        color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='tenure_months', y='monthly_charges',
                           color='Risk_Level',
                           title="Risk Level by Tenure and Monthly Charges",
                           color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()