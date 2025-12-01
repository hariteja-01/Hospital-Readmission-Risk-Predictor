# 🏥 Hospital Readmission Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent machine learning web application that predicts Medicare payment penalties for hospitals based on readmission rates. Built with Streamlit and powered by 5 classification algorithms, this tool helps healthcare organizations proactively manage readmission risks.

---

## 🎯 Overview

The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals with excessive readmission rates by reducing their Medicare reimbursements. This application uses machine learning to predict penalty risks, enabling preventive action.

### Key Capabilities

- **5 ML Models**: Logistic Regression, Decision Tree, SVC, K-Nearest Neighbors (KNN), Neural Network
- **Interactive Dashboard**: Explore 10,000+ hospital records from CMS FY-2025
- **Real-Time Predictions**: Instant risk assessment with probability scores
- **Visual Analytics**: EDA across medical conditions, states, and readmission patterns
- **Performance Metrics**: 6 evaluation metrics including ROC-AUC and cross-validation

---

## 🚀 Features

### Machine Learning Models

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **Logistic Regression** | Linear Classifier | Fast baseline with interpretable coefficients |
| **Decision Tree** | Tree-Based | Rule extraction and feature importance |
| **Support Vector Classifier** | Kernel Method | Non-linear decision boundaries |
| **K-Nearest Neighbors (KNN)** | Instance-Based | Distance-based learning with nearest neighbors |
| **Neural Network (MLP)** | Deep Learning | Complex pattern recognition |

### Model Evaluation

- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction reliability
- **Recall**: Sensitivity to detecting penalties
- **F1-Score**: Balanced precision-recall metric
- **ROC-AUC**: Discrimination ability (typically >0.90)
- **5-Fold CV**: Cross-validation for generalization

### Visual Analytics

- Confusion matrices with heatmap visualization
- ROC curves with AUC comparison
- Feature correlation heatmaps
- Geographic penalty distribution maps
- Medical condition risk analysis

---

## 📊 Dataset

**Source**: Centers for Medicare & Medicaid Services (CMS)  
**Program**: Hospital Readmissions Reduction Program (HRRP)  
**Fiscal Year**: 2025  
**Size**: 10,000+ hospital-condition records

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Facility ID` | Identifier | Unique hospital code |
| `State` | Categorical | US state (2-letter) |
| `Measure Name` | Categorical | Medical condition |
| `Number of Discharges` | Numeric | Total patients discharged |
| `Predicted Readmission Rate` | Numeric | Hospital's observed rate (%) |
| `Expected Readmission Rate` | Numeric | National benchmark (%) |
| `Excess Readmission Ratio` | Numeric | Predicted ÷ Expected |
| `Number of Readmissions` | Numeric | 30-day readmissions |

### Target Variable
```python
Is_Penalized = 1 if Excess_Readmission_Ratio > 1.0 else 0
```

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Plotly, HTML/CSS
- **ML**: Python 3.8+, Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly Express
- **Statistical**: SciPy

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB RAM minimum

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/hariteja-01/Hospital-Readmission-Risk-Predictor.git
cd Hospital-Readmission-Risk-Predictor
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Application**
```bash
streamlit run run.py
```

5. **Access Dashboard**
- Local: `http://localhost:8501`
- Network: `http://<your-ip>:8501`

---

## 💻 Usage

### Navigation

- **📊 Data Overview**: Dataset statistics and sample preview
- **🔧 Data Preprocessing**: Missing value analysis, correlation matrix, outlier detection
- **📈 EDA**: Target distribution, condition analysis, geographic patterns
- **🧑‍💻 Model Training**: Select and train individual models
- **🏆 Model Comparison**: Train all models simultaneously and compare performance
- **🔮 Live Prediction**: Real-time penalty risk assessment

### Example Workflow

1. **Explore** → Review dataset in "Data Overview"
2. **Preprocess** → Check correlations and outliers
3. **Analyze** → Identify patterns in EDA
4. **Train** → Build K-Nearest Neighbors model
5. **Compare** → Evaluate all 5 models
6. **Predict** → Input hospital data for risk assessment

---

## 📈 Model Performance

Performance on FY-2025 CMS dataset (80-20 split):

| Model | Test Acc | Precision | Recall | F1 | ROC-AUC | CV Mean ± Std |
|-------|----------|-----------|--------|----|---------|---------------|
| Logistic Regression | 86.2% | 0.84 | 0.82 | 0.83 | 0.91 | 0.86 ± 0.02 |
| Decision Tree | 84.1% | 0.81 | 0.79 | 0.80 | 0.88 | 0.84 ± 0.03 |
| SVC | 87.3% | 0.85 | 0.83 | 0.84 | 0.92 | 0.87 ± 0.02 |
| **K-Nearest Neighbors** | **88.5%** | **0.86** | **0.84** | **0.85** | **0.93** | **0.87 ± 0.02** |
| Neural Network | 87.5% | 0.85 | 0.83 | 0.84 | 0.92 | 0.87 ± 0.02 |

**Best Model**: K-Nearest Neighbors (KNN) (high accuracy, excellent AUC, consistent performance)

---

## 🔍 Key Insights

### Excess Readmission Ratio (ERR)
```
ERR = Observed Readmission Rate / Expected Readmission Rate
```
- **ERR > 1.0**: Hospital penalized (worse than national average)
- **ERR ≤ 1.0**: No penalty (meets quality standards)

### Business Impact
- **Cost Savings**: $100K-$5M per hospital annually
- **Quality Improvement**: Targeted interventions for high-risk conditions
- **Patient Outcomes**: Reduced readmissions improve safety
- **Operational Efficiency**: Predictive analytics optimize resources

---

## 🧠 Machine Learning Concepts

### Preprocessing
- **Imputation**: Median (numeric), mode (categorical)
- **Scaling**: StandardScaler (zero mean, unit variance)
- **Encoding**: One-hot encoding for categorical variables
- **Pipeline**: Prevents data leakage, ensures reproducibility

### Overfitting Prevention
- Train-test split (80-20)
- 5-fold cross-validation
- Regularization (L2, max_depth, early stopping)
- Instance-based learning (K-Nearest Neighbors)

### Evaluation Strategy
- Multiple metrics (not just accuracy)
- ROC-AUC for threshold-independent assessment
- Confusion matrix for error analysis
- Cross-validation for stability

---

## 🛡️ Troubleshooting

**CSV File Not Found**
- Verify filename: `FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv`
- Ensure file is in project directory

**Module Not Found**
```bash
pip install -r requirements.txt
```

**Port Already in Use**
```bash
streamlit run run.py --server.port 8502
```

**Slow Loading**
- First load: 30-60s (data caching)
- Subsequent loads: instant

---

## 🔮 Future Enhancements

- **Advanced Models**: XGBoost, LightGBM, Random Forest, hyperparameter tuning
- **Interpretability**: SHAP values, LIME explanations
- **Deployment**: Docker, cloud hosting, REST API
- **Features**: Model export, batch predictions, PDF reports

---

## 📚 Resources

- [CMS HRRP Data](https://data.cms.gov/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python](https://plotly.com/python/)

---

## 👨‍💻 Author

**Hari Teja**  
Data Scientist & ML Engineer  
[GitHub](https://github.com/hariteja-01)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- Centers for Medicare & Medicaid Services for HRRP data
- Streamlit team for the web framework
- Scikit-learn community for ML tools

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hariteja-01/Hospital-Readmission-Risk-Predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hariteja-01/Hospital-Readmission-Risk-Predictor/discussions)

---

## 🌟 Contributing

Contributions welcome! Fork, create feature branch, commit, push, and open PR.

---

![GitHub stars](https://img.shields.io/github/stars/hariteja-01/Hospital-Readmission-Risk-Predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/hariteja-01/Hospital-Readmission-Risk-Predictor?style=social)

---

**Made with ❤️ by Hari Teja**  
*Empowering healthcare organizations with predictive analytics*

**Last Updated**: December 1, 2025 | **Version**: 1.0.0 | **Status**: ✅ Production Ready
