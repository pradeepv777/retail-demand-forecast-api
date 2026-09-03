# Retail Demand Forecasting and Promotion Analysis

An end-to-end data science project using the Rossmann Store Sales dataset to analyze retail demand patterns, compare forecasting approaches, examine promotion impact, and serve predictions via a production-ready API.

## Project Overview

Retail businesses need accurate demand forecasts to support inventory planning, staffing, and operational decisions. This project analyzes historical sales data from Rossmann stores and answers two main questions:

1. How accurately can future retail demand be forecasted using classical time series models and machine learning?
2. How are promotions associated with sales after accounting for observable store and calendar differences?

The project follows a complete workflow from exploratory analysis through model evaluation, statistical analysis, and deployment as a containerized REST API.

---

## Key Results

### Forecasting Performance (chain-wide, 60-day holdout)

| Rank | Model | MAE | RMSE |
|---|---|---:|---:|
| 1 | LightGBM | **323,834** | **451,189** |
| 2 | SARIMA | 1,122,513 | 1,540,606 |
| 3 | ARIMA | 2,075,115 | 2,727,432 |
| 4 | Seasonal Naive | 2,084,718 | 2,570,396 |
| 5 | Naive | 2,746,029 | 4,278,452 |

LightGBM reduced MAE by **71.15%** and RMSE by **70.71%** over the best classical model (SARIMA).

### Per-Store Forecasting Performance (60-day holdout)

A separate store-level LightGBM model was trained on individual store-day rows with store metadata features:

- Test MAE: **~659 per store per day**
- Test RMSPE: **~14.4%**

### Promotion Analysis

- Average sales without promotion: **5,929**
- Average sales with promotion: **8,228**
- Unadjusted lift: **+2,299 (+38.77%)**
- OLS regression (controlling for store, day of week, holidays): **+2,293.66** (p < 0.0001, 95% CI [2,287, 2,301])
- Stores with higher sales during promotions: **1,114 out of 1,115**

---

## API

The trained models are served as a REST API built with FastAPI and containerized with Docker.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check + model status |
| POST | `/forecast` | Chain-wide daily sales forecast (all 1,115 stores) |
| POST | `/forecast/store` | Per-store daily sales forecast |
| POST | `/forecast/store/whatif` | Promo vs no-promo comparison for a single store |

Full interactive documentation available at `http://localhost:8000/docs` (Swagger UI).

### Running locally

```bash
docker-compose up --build
```

Then open `http://localhost:8000/docs`.

### Example requests

**Chain-wide forecast:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

**Per-store forecast:**
```bash
curl -X POST http://localhost:8000/forecast/store \
  -H "Content-Type: application/json" \
  -d '{"store_id": 42, "days": 7}'
```

**What-if promo scenario:**
```bash
curl -X POST http://localhost:8000/forecast/store/whatif \
  -H "Content-Type: application/json" \
  -d '{"store_id": 42, "days": 7}'
```

> On Windows PowerShell use `Invoke-WebRequest` with `-Body` instead of curl single-quote syntax.

---

## Project Workflow

### 1. Exploratory Data Analysis
Notebook: [`01_eda.ipynb`](notebooks/01_eda.ipynb)

Sales distributions, open/closed patterns, day-of-week seasonality, monthly and yearly trends, promotion behavior, holiday effects, and customer-sales relationships.

### 2. Classical Time Series Forecasting
Notebook: [`02_forecasting_classical.ipynb`](notebooks/02_forecasting_classical.ipynb)

Naive, Seasonal Naive, ARIMA, and SARIMA models evaluated on a chronological 60-day holdout. SARIMA was the strongest classical model.

### 3. Machine Learning Forecasting
Notebook: [`03_forecasting_modern.ipynb`](notebooks/03_forecasting_modern.ipynb)

LightGBM trained on lag features, rolling averages, and calendar features on chain-wide aggregated daily data. Substantially outperformed all classical models.

### 4. Model Comparison
Notebook: [`04_model_comparison.ipynb`](notebooks/04_model_comparison.ipynb)

Fair comparison across all models on the same target and test period. LightGBM selected as the best model.

### 5. Promotion Analysis
Notebook: [`05_promotion_analysis.ipynb`](notebooks/05_promotion_analysis.ipynb)

Promotion vs non-promotion sales comparison with within-store controls, OLS regression, confidence intervals, and discussion of observational limitations.

### 6. Per-Store Forecasting
Notebook: [`06_store_level_forecasting.ipynb`](notebooks/06_store_level_forecasting.ipynb)

Store-level LightGBM trained on individual store-day rows with store metadata features (StoreType, Assortment, CompetitionDistance, Promo2). Enables per-store demand forecasting and powers the `/forecast/store` API endpoints.

### 7. A/B Test — Promotion Effect
Notebook: [`07_ab_test_promo_effect.ipynb`](notebooks/07_ab_test_promo_effect.ipynb)

Welch's t-test on 844,392 open store-day observations. Promo lift of ~2,299/day is statistically significant (p < 0.0001) and consistent across all four store types. Power analysis, stability check, and causal inference caveats included.

---

## Dataset

The project uses the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset.

| File | Description |
|---|---|
| `train.csv` | Historical store-level daily sales (1,017,209 rows) |
| `store.csv` | Store metadata — type, assortment, competition, promotions |
| `test.csv` | Test data without sales values |

---

## Repository Structure

```text
rossmann_demand_forecasting/
│
├── api/
│   ├── main.py          # FastAPI app — all endpoints
│   └── schemas.py       # Pydantic request/response models
│
├── Data/
│   └── raw/
│       ├── train.csv
│       ├── store.csv
│       └── test.csv
│
├── figures/             # Saved plots from notebooks
├── models/
│   ├── lightgbm_model.pkl        # Chain-wide forecast model
│   ├── lightgbm_store_model.pkl  # Per-store forecast model
│   └── store_features.pkl        # Encoded store metadata lookup
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_forecasting_classical.ipynb
│   ├── 03_forecasting_modern.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_promotion_analysis.ipynb
│   ├── 06_store_level_forecasting.ipynb
│   └── 07_ab_test_promo_effect.ipynb
│
├── reports/
│   └── insights.txt     # Full analytical findings
│
├── tests/
│   └── test_smoke.py    # Smoke tests (run against live container)
│
├── .dockerignore
├── .github/
│   └── workflows/
│       ├── ci.yml       # Build + smoke test on every push
│       └── cd.yml       # Deploy to EC2 on push to main
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

---

## Deployment (EC2)

The CD pipeline in `.github/workflows/cd.yml` builds the Docker image and deploys it to an EC2 instance on every push to `main`.

Before the first deployment, upload the model files and data to EC2 once:

```bash
scp -i your-key.pem models/lightgbm_model.pkl       ec2-user@<host>:~/rossmann/models/
scp -i your-key.pem models/lightgbm_store_model.pkl ec2-user@<host>:~/rossmann/models/
scp -i your-key.pem models/store_features.pkl       ec2-user@<host>:~/rossmann/models/
scp -i your-key.pem Data/raw/train.csv              ec2-user@<host>:~/rossmann/Data/raw/
scp -i your-key.pem Data/raw/store.csv              ec2-user@<host>:~/rossmann/Data/raw/
```

Required GitHub Secrets: `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`.
