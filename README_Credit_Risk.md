# German Credit Risk Classification with SMOTE

Machine learning classification system for credit risk assessment using multiple algorithms with advanced class imbalance handling techniques.

## 🎯 Project Overview

This project implements and compares multiple machine learning classifiers (Perceptron, Logistic Regression, SVM, MLP) to predict credit risk using the German Credit dataset. The project focuses on handling class imbalance through SMOTENC and feature engineering using polynomial features.

## 📊 Dataset

- **Source**: UCI Machine Learning Repository - Statlog (German Credit Data)
- **URL**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- **Size**: 1,000 credit records
- **Features**: 20 attributes (7 numerical, 13 categorical)
- **Target**: Binary classification (Good Credit = 1, Bad Credit = 0)
- **Class Distribution**: Imbalanced (70% Good, 30% Bad)

### Key Features
- **Numerical**: Duration, CreditAmount, Age, InstallmentRate, ExistingCredits, ResidenceYears, LiablePersons
- **Categorical**: Status, CreditHistory, Purpose, Savings, Employment, Sex, Debtors, Property, OtherPlans, Housing, Job, Telephone, ForeignWorker

## 🏗️ Methodology

### R1: Data Preprocessing
```python
1. Imputation: Median strategy for missing numerical values
2. Normalization: MinMaxScaler for numerical features
3. Feature Engineering: Polynomial features (degree=2, interactions only)
4. Cross-Validation: RepeatedStratifiedKFold (10 splits, 3 repeats)
```

### R2: Model Comparison
Four classifiers compared using 10-fold repeated cross-validation:

| Model | Accuracy | F1-Score | Algorithm |
|-------|----------|----------|-----------|
| Perceptron | 0.607 | 0.663 | SGDClassifier(loss='perceptron') |
| Logistic Regression | 0.677 | 0.781 | SGDClassifier(loss='log_loss') |
| SVM | 0.682 | 0.790 | SGDClassifier(loss='hinge') |
| **MLP** | **0.700** | **0.811** | **MLPClassifier(hidden_layer_sizes=(100,))** |

**Winner**: MLP Neural Network with single hidden layer (100 neurons)

### R3: Class Imbalance Handling

Comparison of two techniques for addressing class imbalance:

#### Class Weighting
```python
SGDClassifier(loss='log_loss', class_weight='balanced')
```
- **Accuracy**: 0.567
- **F1-Score**: 0.601

#### SMOTENC (Synthetic Minority Over-sampling)
```python
SMOTENC(categorical_features=categorical_feature_indices)
```
- **Accuracy**: 0.600
- **F1-Score**: 0.677

**Result**: SMOTENC outperformed class weighting by 7.6% in F1-score

### R4: Feature Selection

Feature importance analysis using logistic regression coefficients:

#### Top 5 Most Important Features
1. **Age** - Customer age
2. **CreditAmount** - Loan amount requested
3. **Duration** - Loan duration in months
4. **ExistingCredits** - Number of existing credits
5. **InstallmentRate** - Installment rate as percentage of income

#### Performance Impact
| Step | F1-Score | Improvement |
|------|----------|-------------|
| Before Feature Selection | 0.601 | - |
| After Feature Selection | 0.831 | +38.3% |

## 📈 Final Results

### Best Model Configuration
```python
SGDClassifier(
    loss='hinge',        # SVM
    alpha=0.001,         # L2 regularization
    max_iter=1000,       # Maximum iterations
    random_state=42
)
```

Found through GridSearchCV with parameters:
- `loss`: ['log_loss', 'hinge']
- `alpha`: [0.0001, 0.001, 0.01, 0.1]
- `max_iter`: [1000, 2000]

### Performance Metrics
- **Test Accuracy**: 0.600
- **Test F1-Score**: 0.831
- **Cross-Validation Accuracy**: 0.682 ± 0.031
- **Cross-Validation F1-Score**: 0.790 ± 0.028

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** - Machine learning algorithms
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Imbalanced-learn** - SMOTENC implementation
- **SciPy** - Statistical analysis

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Devotion25/German-Credit-Risk-Analysis.git

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```python
# Open Jupyter notebook
jupyter notebook credit_risk_analysis.ipynb

# Or run in Google Colab
# Upload the notebook to Colab and run all cells
```

## 📊 Visualizations

The project includes:
1. **Feature Distribution Analysis**: Histograms and box plots
2. **Correlation Heatmap**: Feature relationships
3. **Model Comparison Charts**: Accuracy and F1-score comparison
4. **Confusion Matrices**: Classification performance
5. **Feature Importance Plot**: Top predictive features
6. **Learning Curves**: Model performance over iterations

## 🔍 Key Insights

1. **Neural Networks Outperform**: MLP achieved highest performance (70% accuracy, 0.811 F1)
2. **SMOTENC Effectiveness**: Synthetic sampling improved F1-score by 7.6% over class weighting
3. **Feature Selection Impact**: Reducing features from 20 to 5 improved F1-score by 38.3%
4. **Critical Features**: Age, Credit Amount, and Duration are strongest predictors of credit risk
5. **Polynomial Features**: Interaction terms provided marginal improvement

## 💡 Real-World Applications

This credit risk model can be applied to:
- **Automated Loan Approval**: Streamline credit decisions
- **Risk Management**: Identify high-risk applicants
- **Portfolio Optimization**: Balance risk across loan portfolio
- **Regulatory Compliance**: Fair and transparent credit scoring
- **Fraud Detection**: Identify suspicious credit applications

## ⚠️ Limitations & Future Work

### Current Limitations
- Small dataset (1,000 records)
- Class imbalance persists even after SMOTENC
- Limited to German credit market (1990s data)
- No temporal validation (single time point)

### Future Improvements
1. **Ensemble Methods**: Random Forest, XGBoost, AdaBoost
2. **Deep Learning**: Deep neural networks with dropout
3. **Temporal Analysis**: Model performance over time
4. **External Validation**: Test on other credit datasets
5. **Explainability**: SHAP values for model interpretation
6. **Real-time Deployment**: REST API for production use

## 🎓 Course Information

- **Course**: CSCE 5215 - Machine Learning
- **Institution**: University of North Texas
- **Semester**: Spring 2025

## 👨‍💻 Author

**Devotion Ekueku**
- GitHub: [@Devotion25](https://github.com/Devotion25)
- LinkedIn: [devotionekueku](https://www.linkedin.com/in/devotionekueku/)

## 📄 License

This project is for educational purposes as part of coursework at the University of North Texas.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset
- Imbalanced-learn library developers
- Course instructors at UNT
- Scikit-learn community

## 📚 References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.
3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
4. Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox. JMLR.
