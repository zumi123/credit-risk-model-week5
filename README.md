# Credit Risk Probability Model — Alternative Data

This project develops a credit scoring model using behavioral transaction data from an eCommerce platform. The objective is to classify customers into high or low credit risk and estimate risk probabilities to guide lending decisions.

## Project Overview

Bati Bank is partnering with an eCommerce platform to offer Buy-Now-Pay-Later services. Since traditional credit scoring features (like income or bureau score) are unavailable, this model leverages **alternative data** such as transaction Recency, Frequency, and Monetary (RFM) behavior to predict customer risk levels.

The model outputs:
- A binary label indicating high or low credit risk
- A risk probability score
- A recommended loan amount and duration

This end-to-end solution includes model training, deployment with FastAPI, and CI/CD using GitHub Actions and Docker.

---

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml        # GitHub Actions workflow for CI
├── data/                           # Raw and processed data
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering pipeline
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Inference logic
│   └── api/
│       ├── main.py                 # FastAPI app
│       └── pydantic_models.py      # Request/response models
├── tests/
│   └── test_data_processing.py     # Unit tests
├── Dockerfile                      # Docker container spec
├── docker-compose.yml              # Docker orchestration
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignored files
└── README.md                       # Project documentation
```

---

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability
Basel II emphasizes **quantitative risk measurement** and mandates that banks justify their internal risk rating systems. As a result, credit scoring models must be **interpretable**, well-documented, and auditable to meet compliance requirements. This ensures regulators and internal stakeholders can trust and validate the models, especially in loan approvals or denials.

### 2. Why Use a Proxy Variable?
In our dataset, there is **no direct "default" label**. Therefore, we engineer a **proxy target** using customer behavior (RFM analysis) to simulate default risk. While necessary for supervised learning, this introduces **business risks**:
- Potential misclassification (label noise)
- Biased or unrepresentative model outcomes
- Legal and ethical implications if misused

To mitigate this, we ensure transparent methodology and rigorous validation of the proxy.

### 3. Simple vs. Complex Models in a Regulated Context
| Criteria                  | Logistic Regression (with WoE)   | Gradient Boosting (e.g., XGBoost) |
|--------------------------|----------------------------------|-----------------------------------|
| Interpretability         | High                             | Low without SHAP/LIME             |
| Regulatory Acceptance    | Strong                           | Requires explanation              |
| Model Performance        | Moderate                         | High                              |
| Deployment Simplicity    | Easy                             | More complex                      |
| Risk of Overfitting      | Lower                            | Higher                            |

In regulated domains, a **simple interpretable model** is often preferred for compliance. Complex models may be adopted when performance gains justify the additional scrutiny and explainability techniques.

---
## Exploratory Data Analysis (EDA)

### Key EDA Insights

Based on the exploratory data analysis of the Xente transactions dataset:

1. **No Missing Data – But Redundant Fields**  
   All columns are fully populated. However, `CurrencyCode` and `CountryCode` have only one unique value each and offer no predictive value. These should be dropped in modeling.

2. **Extreme Class Imbalance in `FraudResult`**  
   Only 0.2% of the data is labeled as fraudulent. Any direct modeling of this column will require resampling or class-weighted techniques to avoid bias toward the majority class.

3. **Highly Skewed Transaction Amounts**  
   The `Amount` feature spans a wide range and is highly skewed with extreme outliers. A log transformation or robust scaling is needed before feeding this into machine learning models.

4. **`Value` is a Redundant Feature**  
   The `Value` column is the absolute value of `Amount` and has near-perfect correlation with it. One of these features should be dropped to avoid multicollinearity.

5. **Few Entities Dominate the Data**  
   A small subset of accounts and product providers contribute disproportionately to transaction volume. This highlights the need for aggregation and normalization in feature engineering to prevent bias from power users.

These findings inform our feature engineering, modeling strategy, and choice of evaluation metrics.

---

## References
- [Basel II Accord Summary](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Credit Risk Concepts – Investopedia](https://www.investopedia.com/terms/c/creditrisk.asp)
- [HKMA on Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Scorecard Modeling Guide](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)

---

