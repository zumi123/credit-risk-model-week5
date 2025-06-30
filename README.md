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

## References
- [Basel II Accord Summary](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Credit Risk Concepts – Investopedia](https://www.investopedia.com/terms/c/creditrisk.asp)
- [HKMA on Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Scorecard Modeling Guide](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)

---

