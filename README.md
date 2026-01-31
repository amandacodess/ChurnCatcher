# ChurnCatcher 🎯

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning system that catches customers before they churn! This project uses advanced analytics and multiple ML models to predict customer churn and help businesses take proactive retention measures.

## 📊 Project Overview

ChurnCatcher analyzes customer behavior patterns to identify those at risk of leaving. The system provides:

- **Exploratory Data Analysis (EDA)** with visualizations
- **Data preprocessing** and feature engineering
- **Multiple ML models** comparison (Logistic Regression, Random Forest, Gradient Boosting)
- **Model evaluation** with comprehensive metrics
- **Feature importance** analysis

## 🎯 Features

- Synthetic customer data generation for demonstration
- Visual analysis of churn patterns
- Automated model selection based on performance
- ROC curves and confusion matrices
- Feature importance rankings

## 📁 Project Structure
```
ChurnCatcher/
│
├── churn_prediction.py          # Main analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── customer_churn_data.csv       # Generated sample data (after running)
├── churn_eda.png                 # EDA visualizations (after running)
├── model_evaluation.png          # Model performance charts (after running)
└── feature_importance.png        # Feature rankings (after running)
```

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/ChurnCatcher.git
   cd ChurnCatcher
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the analysis**
```bash
   python churn_prediction.py
```

## 📈 Results

ChurnCatcher generates:

1. **customer_churn_data.csv** - Sample dataset with 5,000 customer records
2. **churn_eda.png** - Exploratory data analysis visualizations
3. **model_evaluation.png** - Model performance comparison
4. **feature_importance.png** - Ranking of features by predictive power

## 🔍 Key Insights

The analysis typically reveals:

- **Contract type** is a strong predictor (month-to-month contracts show higher churn)
- **Tenure** negatively correlates with churn (newer customers more likely to leave)
- **Customer service calls** can indicate dissatisfaction
- **Monthly charges** impact retention rates

## 🤖 Models Used

- **Logistic Regression** - Baseline linear model
- **Random Forest** - Ensemble method with feature importance
- **Gradient Boosting** - Advanced ensemble technique

The best model is automatically selected based on AUC-ROC score.

## 📊 Sample Metrics

Typical model performance:
- Accuracy: 75-85%
- AUC-ROC: 0.80-0.90
- Precision/Recall: Balanced for both classes

## 🛠️ Customization

You can modify the analysis by:

- Adjusting sample size in `generate_sample_data(n_samples=5000)`
- Adding new features in the data generation process
- Testing additional ML algorithms
- Tuning hyperparameters for better performance

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Want to try it live?

https://churnchaser-project.streamlit.app/

---

**ChurnCatcher** - Catch customers before they leave! 🎯
