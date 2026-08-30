# Retail Demand Forecasting and Promotion Analysis

An end-to-end data science project using the Rossmann Store Sales dataset to analyze retail demand patterns, compare forecasting approaches, and examine the relationship between promotions and sales.

## Project Overview

Retail businesses need accurate demand forecasts to support inventory planning, staffing, and operational decisions. This project analyzes historical sales data from Rossmann stores and answers two main questions:

1. How accurately can future retail demand be forecasted using classical time series models and machine learning?
2. How are promotions associated with sales after accounting for observable store and calendar differences?

The project follows a complete workflow from exploratory analysis to model evaluation and statistical analysis.

## Key Results

### Forecasting Performance

All forecasting models were evaluated on the same chronological 60-day test period.

| Rank | Model | MAE | RMSE |
|---|---|---:|---:|
| 1 | LightGBM | **323,834** | **451,189** |
| 2 | SARIMA | 1,122,513 | 1,540,606 |
| 3 | ARIMA | 2,075,115 | 2,727,432 |
| 4 | Seasonal Naive | 2,084,718 | 2,570,396 |
| 5 | Naive | 2,746,029 | 4,278,452 |

LightGBM achieved the lowest error across both MAE and RMSE.

Compared with the best classical model, SARIMA, LightGBM reduced:

- MAE by **71.15%**
- RMSE by **70.71%**

### Promotion Analysis

The promotion analysis was performed using only days when stores were open.

- Average sales without promotion: **5,929**
- Average sales with promotion: **8,228**
- Unadjusted difference: **2,299**
- Relative difference: **38.77%**
- Stores with higher average sales during promotions: **1,114 out of 1,115**

An OLS regression controlling for store, day of week, school holidays, and state holidays estimated a promotion association of approximately:

**+2,293.66 in average sales**

This result is interpreted as an association, not a causal effect, because the dataset is observational and unobserved confounding may remain.

---

# Project Workflow

## 1. Exploratory Data Analysis

Notebook: [`01_eda.ipynb`](notebooks/01_eda.ipynb)

The exploratory analysis investigates:

- Sales distributions
- Open and closed store patterns
- Day-of-week seasonality
- Monthly and yearly sales patterns
- Promotion behavior
- Holiday effects
- Customer and sales relationships

The analysis identified strong weekly patterns in retail demand, which motivated the use of seasonal forecasting models.

---

## 2. Classical Time Series Forecasting

Notebook: [`02_forecasting_classical.ipynb`](notebooks/02_forecasting_classical.ipynb)

Classical forecasting models were developed using aggregated daily Rossmann sales.

Models evaluated:

- Naive Forecast
- Seasonal Naive Forecast
- ARIMA
- SARIMA

A chronological train-test split was used to avoid data leakage.

SARIMA was the strongest classical forecasting model and captured the weekly seasonal structure better than the non-seasonal models.

---

## 3. Machine Learning Forecasting

Notebook: [`03_forecasting_modern.ipynb`](notebooks/03_forecasting_modern.ipynb)

A LightGBM regression model was trained using time-based features.

Features include:

- Lagged sales
- Rolling averages
- Day of week
- Day of month
- Month
- Promotion-related features
- Other calendar information

The model was evaluated on the same chronological test period used for the classical models.

LightGBM substantially outperformed the classical forecasting approaches.

---

## 4. Model Comparison

Notebook: [`04_model_comparison.ipynb`](notebooks/04_model_comparison.ipynb)

All forecasting models were compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The comparison uses the same target and test period across models to ensure a fair evaluation.

LightGBM was selected as the best-performing model based on both evaluation metrics.

---

## 5. Promotion Analysis

Notebook: [`05_promotion_analysis.ipynb`](notebooks/05_promotion_analysis.ipynb)

The promotion analysis examines the relationship between promotional activity and sales.

The analysis includes:

- Promotion versus non-promotion sales comparison
- Within-store comparison
- Calendar-controlled comparisons
- OLS regression
- Confidence intervals
- Discussion of confounding and observational limitations

Only open-store observations are used in the analysis to avoid zero-sales values from closed stores distorting the comparison.

The results consistently show higher sales during promotion-active periods. However, the analysis does not claim that promotions directly caused the observed sales increase.

---

# Dataset

The project uses the Rossmann Store Sales dataset.

Main files:

- `train.csv` - Historical store-level daily sales data
- `test.csv` - Test data without sales values
- `store.csv` - Store-level information

Important variables include:

| Variable | Description |
|---|---|
| Store | Store identifier |
| Date | Date of observation |
| Sales | Daily store sales |
| Customers | Number of customers |
| Open | Whether the store was open |
| Promo | Whether a promotion was active |
| StateHoliday | State holiday indicator |
| SchoolHoliday | School holiday indicator |

---

# Repository Structure

```text
retail_demand_forecasting/
│
├── Data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── store.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_forecasting_classical.ipynb
│   ├── 03_forecasting_modern.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_promotion_analysis.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md