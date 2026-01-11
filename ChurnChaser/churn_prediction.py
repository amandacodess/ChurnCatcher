import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
    def generate_sample_data(self, n_samples=5000):
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
        
        # Create churn with realistic patterns
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
    
    def exploratory_analysis(self, df):
        """Perform EDA and create visualizations"""
        print("=" * 60)
        print("CUSTOMER CHURN ANALYSIS - EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        print("\n1. Dataset Overview:")
        print(f"   Total customers: {len(df)}")
        print(f"   Churned customers: {df['churn'].sum()} ({df['churn'].mean()*100:.2f}%)")
        print(f"   Retained customers: {(1-df['churn']).sum()} ({(1-df['churn'].mean())*100:.2f}%)")
        
        print("\n2. Missing Values:")
        print(df.isnull().sum())
        
        print("\n3. Numerical Features Statistics:")
        print(df.describe())
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Customer Churn Analysis - Visual Insights', fontsize=16, fontweight='bold')
        
        # Churn distribution
        churn_counts = df['churn'].value_counts()
        axes[0, 0].pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Overall Churn Distribution')
        
        # Churn by contract type
        pd.crosstab(df['contract_type'], df['churn'], normalize='index').plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Churn Rate by Contract Type')
        axes[0, 1].set_xlabel('Contract Type')
        axes[0, 1].set_ylabel('Proportion')
        axes[0, 1].legend(['Retained', 'Churned'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Tenure distribution
        df.groupby('churn')['tenure_months'].plot(kind='hist', ax=axes[0, 2], alpha=0.6, bins=20, legend=True)
        axes[0, 2].set_title('Tenure Distribution by Churn')
        axes[0, 2].set_xlabel('Tenure (months)')
        axes[0, 2].legend(['Retained', 'Churned'])
        
        # Monthly charges
        df.boxplot(column='monthly_charges', by='churn', ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Charges by Churn Status')
        axes[1, 0].set_xlabel('Churn (0=No, 1=Yes)')
        axes[1, 0].set_ylabel('Monthly Charges ($)')
        plt.sca(axes[1, 0])
        plt.xticks([1, 2], ['Retained', 'Churned'])
        
        # Customer service calls
        pd.crosstab(df['customer_service_calls'], df['churn'], normalize='index').plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
        axes[1, 1].set_title('Churn Rate by Service Calls')
        axes[1, 1].set_xlabel('Number of Service Calls')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].legend(['Retained', 'Churned'])
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 2], cbar_kws={'shrink': 0.8})
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('churn_eda.png', dpi=300, bbox_inches='tight')
        print("\n4. Visualizations saved as 'churn_eda.png'")
        plt.show()
        
    def preprocess_data(self, df):
        """Prepare data for modeling"""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Drop customer_id
        df_processed = df.drop('customer_id', axis=1)
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        print(f"\nEncoded {len(categorical_cols)} categorical features")
        
        # Split features and target
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 60)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC-ROC: {auc_score:.4f}")
        
        # Select best model based on AUC
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.model = results[best_model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        print(f"{'='*60}")
        
        return results, best_model_name
    
    def evaluate_model(self, results, best_model_name, y_test):
        """Create detailed evaluation visualizations"""
        best_result = results[best_model_name]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Model Evaluation: {best_model_name}', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticklabels(['Retained', 'Churned'])
        axes[0].set_yticklabels(['Retained', 'Churned'])
        
        # ROC Curve
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            axes[1].plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.3f})")
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Model Comparison
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        aucs = [results[m]['auc'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[2].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        axes[2].bar(x + width/2, aucs, width, label='AUC-ROC', color='lightcoral')
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Model Performance Comparison')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, rotation=15, ha='right')
        axes[2].legend()
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nEvaluation visualizations saved as 'model_evaluation.png'")
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, best_result['predictions'], target_names=['Retained', 'Churned']))
    
    def feature_importance(self, feature_names):
        """Display feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("=" * 60)
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance for Churn Prediction', fontweight='bold', fontsize=14)
            plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
            plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
            plt.ylabel('Importance Score')
            plt.xlabel('Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance chart saved as 'feature_importance.png'")
            plt.show()
            
            print("\nTop 5 Most Important Features:")
            for i in range(min(5, len(importances))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Generate sample data
    print("\nGenerating sample customer data...")
    df = predictor.generate_sample_data(n_samples=5000)
    
    # Save raw data
    df.to_csv('customer_churn_data.csv', index=False)
    print("Sample data saved as 'customer_churn_data.csv'")
    
    # Exploratory analysis
    predictor.exploratory_analysis(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = predictor.preprocess_data(df)
    
    # Train models
    results, best_model_name = predictor.train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate best model
    predictor.evaluate_model(results, best_model_name, y_test)
    
    # Feature importance
    predictor.feature_importance(feature_names)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - customer_churn_data.csv (raw data)")
    print("  - churn_eda.png (exploratory analysis)")
    print("  - model_evaluation.png (model performance)")
    print("  - feature_importance.png (feature rankings)")
    
if __name__ == "__main__":
    main()