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

## Feature Engineering

I built a modular and reproducible preprocessing pipeline using `scikit-learn` to convert raw transaction data into model-ready features. The core components of the pipeline include:

1. **Date/Time Feature Extraction** 
   Extracts hour, day, month, and year from the `TransactionStartTime` field using a custom transformer.

2. **Numeric Standardization** 
   Standard scales the `Amount` and `Value` columns to normalize their distributions.

3. **Categorical Encoding** 
   Uses `OneHotEncoder` to transform high-cardinality fields like `ProviderId` and `ProductId` into binary indicator variables. We ensured compatibility with scikit-learn ≥1.2 using the `sparse_output=False` parameter.

4. **Custom Transformer Support** 
   Custom transformers (e.g., `DateTimeFeatures`) were updated to support `.set_output()` to ensure the pipeline can return a DataFrame using `set_output(transform="pandas")`.

5. **Output** 
   The pipeline was successfully applied to a demo dataset of 10,000 rows and returned a transformed matrix with **50 engineered features**. Sample feature names include:

   ```
   ['datetime__tx_hour', 'datetime__tx_day', 'datetime__tx_month', 'datetime__tx_year',
    'numeric__Amount', 'numeric__Value',
    'categorical__ProviderId_ProviderId_1', 'categorical__ProductId_ProductId_10', ...]
   ```

This pipeline is implemented in `src/data_processing.py` and will be used in subsequent tasks for model training and prediction.
---



## Proxy Target Variable Engineering

Because the raw dataset contains **no explicit `default` label**, we created a proxy called `is_high_risk` using Recency–Frequency–Monetary (RFM) analysis:

1. **RFM Computation**  
   * *Recency* = days since a customer’s last transaction (relative to a fixed snapshot date).  
   * *Frequency* = total number of transactions per customer.  
   * *Monetary* = total absolute value of all transactions per customer.

2. **Clustering**  
   * Scaled the RFM features with `StandardScaler`.  
   * Applied **K‑Means** with *n_clusters = 3* and `random_state=42` to obtain three behavioral segments.

3. **High‑Risk Segment Identification**  
   * Calculated cluster centroids and automatically chose the segment with **highest Recency** and **lowest Frequency/Monetary** as the *least engaged* group.  
   * Assigned **`is_high_risk = 1`** to customers in this cluster; all others receive **0**.

4. **Dataset Integration**  
   * Merged the new binary label back into the main DataFrame, making it available for downstream model training (`train.py`).  
   * Saved an intermediate CSV (`data/processed/rfm_labels.csv`) for transparency and QA.

This logic is fully encapsulated in `src/proxy_target.py`, exposing two helpers:

```python
from src.proxy_target import compute_rfm, add_is_high_risk
```

which can be imported anywhere in the pipeline or in unit tests.

---


## Model Training, Tuning, and Tracking

I developed a robust supervised learning pipeline to train models that predict the `is_high_risk` proxy label.

### 1. Model Candidates
I trained and tuned the following classifiers:
- **Logistic Regression**
- **Gradient Boosting Machine (GBM)**

### 2. Data Pipeline
- The dataset was preprocessed using the Task 3 feature pipeline.
- Labels were assigned using Task 4 (`is_high_risk`).
- Data was split 80/20 using `stratify=y` to preserve class balance.

### 3. Hyperparameter Tuning
Each model was wrapped in a `Pipeline` and tuned using `RandomizedSearchCV`.

Best parameters for **GBM**:
```
max_depth: 3
n_estimators: 300
learning_rate: 0.05
```

### 4. Evaluation Metrics
The GBM model outperformed Logistic Regression across all metrics:

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.9998    |
| Precision  | 0.9976    |
| Recall     | 0.9976    |
| F1 Score   | 0.9976    |
| ROC-AUC    | 1.0000    |

### 5. Experiment Tracking
- All experiments, metrics, and artifacts were logged using **MLflow**.
- The best model was automatically **registered** in the **MLflow Model Registry**.

To view results:
```bash
mlflow ui
# then visit http://localhost:5000
```
### 6. Unit Testing

I wrote unit tests for two key helper functions:

- `DateTimeFeatures` transformer (from `src.data_processing`)
- `compute_rfm` function (from `src.proxy_target`)

These are located in:

```bash
tests/test_data_processing.py

```
---

## Model Deployment & CI/CD

### FastAPI Service
- **Location:** `src/api/`
- Loads best model from MLflow via `BEST_MODEL_RUN_ID`.
- `/predict` endpoint returns `risk_probability`.
- Swagger UI at `/docs`.

Run locally:

```bash
export MLFLOW_TRACKING_URI=./mlruns
export BEST_MODEL_RUN_ID=<best_run_id>
uvicorn src.api.main:app --reload --port 8000
```

### Docker
```bash
docker compose build --build-arg RUN_ID=<best_run_id>
docker compose up
```
Container exposes `http://localhost:8000/docs`.

### GitHub Actions CI
Workflow `.github/workflows/ci.yml`:

| Stage | Tool  |
|-------|-------|
| Lint  | flake8|
| Test  | pytest|

Build fails on any lint or test error.

---

## References
- [Basel II Accord Summary](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Credit Risk Concepts – Investopedia](https://www.investopedia.com/terms/c/creditrisk.asp)
- [HKMA on Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Scorecard Modeling Guide](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)

---

