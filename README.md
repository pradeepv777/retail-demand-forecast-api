# Retail Demand Forecasting and Promotion Impact Analysis

## Project Overview

This project analyzes the **Rossmann Store Sales** dataset to understand historical retail sales patterns, develop and evaluate classical time series and machine learning forecasting models, and quantify the association between promotions and sales. 

The project delivers an end-to-end empirical workflow:
- Exploratory data analysis uncovering weekly seasonality, holiday effects, and customer dynamics.
- Classical statistical time series forecasting using Naive, Seasonal Naive, ARIMA, and SARIMA models on aggregated store demand.
- Gradient boosted decision trees (LightGBM) leveraging calendar features, autoregressive lags, and rolling window aggregations.
- Rigorous out-of-sample model comparison on a shared 60-day holdout period.
- Promotion impact analysis controlling for store-level fixed differences and calendar covariates while carefully distinguishing statistical association from causal effects.

---

## Business Problem

Accurate demand forecasting and promotion analytics are foundational to retail supply chain efficiency:
- **Inventory Planning:** Overestimating demand creates excess holding costs and waste; underestimating leads to stockouts and lost revenue.
- **Staffing Optimization:** Accurately predicting weekly peaks and promotional surges ensures optimal store labor scheduling.
- **Supply Chain Coordination:** Reliable lead-time forecasts enable efficient warehouse distribution and logistics planning.
- **Promotional Strategy:** Evaluating promotional uplift helps marketing and commercial teams assess demand sensitivity and baseline performance.

---

## Dataset

The analysis uses the **Rossmann Store Sales** historical dataset spanning over 1,115 drug stores across Germany:
- **Store:** Unique store identifier (1 to 1,115).
- **Date:** Daily record date (2013-01-01 to 2015-07-31).
- **Sales:** Daily turnover in euros (forecast target).
- **Customers:** Number of daily store visitors.
- **Open:** Store operating status (`1` = Open, `0` = Closed).
- **Promo:** Active store promotional discount (`1` = Promotion active, `0` = No promotion).
- **StateHoliday:** Categorical state holiday indicator (`a` = public holiday, `b` = Easter holiday, `c` = Christmas, `0` = None).
- **SchoolHoliday:** Binary indicator for regional public school closures.

---

## Project Workflow

The project is structured into five sequential Jupyter notebooks:

1. **[01_eda.ipynb](file:///c:/Users/prade/OneDrive/Desktop/retail_demand_forecasting/notebooks/01_eda.ipynb) — Exploratory Data Analysis:**
   Investigates sales distributions, strong day-of-week seasonality, Sunday store closures, holiday impacts, customer-to-sales ratios, and competition distance.
2. **[02_forecasting_classical.ipynb](file:///c:/Users/prade/OneDrive/Desktop/retail_demand_forecasting/notebooks/02_forecasting_classical.ipynb) — Classical Time Series Forecasting:**
   Aggregates total daily demand across the retail network and establishes baseline statistical benchmarks (Naive, Seasonal Naive, ARIMA, and seasonal grid-searched SARIMA).
3. **[03_forecasting_modern.ipynb](file:///c:/Users/prade/OneDrive/Desktop/retail_demand_forecasting/notebooks/03_forecasting_modern.ipynb) — Machine Learning Forecasting:**
   Engineers calendar features, autoregressive lag features (`Lag_1`, `Lag_7`, `Lag_14`), rolling window statistics (`Rolling_Mean_7`, `Rolling_Mean_14`), and trains a LightGBM regressor.
4. **[04_model_comparison.ipynb](file:///c:/Users/prade/OneDrive/Desktop/retail_demand_forecasting/notebooks/04_model_comparison.ipynb) — Model Comparison & Selection:**
   Compares classical and modern forecasting architectures on a standardized 60-day chronological test window using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
5. **[05_promotion_analysis.ipynb](file:///c:/Users/prade/OneDrive/Desktop/retail_demand_forecasting/notebooks/05_promotion_analysis.ipynb) — Promotion Impact Analysis:**
   Analyzes promotional lift across open-store observations (`Open == 1`), verifies store-level consistency, controls for calendar confounders via Ordinary Least Squares (OLS) regression, and discusses observational limitations.

---

## Exploratory Data Analysis

Key insights identified during exploratory data analysis:
- **Strong Weekly Seasonality:** Sales peak strongly on Mondays and Fridays, with sharp drops on Sundays due to standard store closures.
- **Customer Volume is the Primary Driver:** Sales scale near-linearly with customer visits, averaging ~€8.94–€9.80 per customer transaction.
- **Promotions Lift Sales and Traffic:** Promotional weekdays exhibit higher customer visits and total revenue compared to non-promotional weekdays.
- **Competition Distance Insufficiency:** Competition proximity alone does not linearly determine sales volume; store format, customer assortment, and location factors dominate.

---

## Forecasting Methodology

- **Target Series:** Aggregated network-wide daily sales.
- **Validation Scheme:** Chronological train-test split:
  - **Train Period:** 2013-01-01 to 2015-06-01 (868 training days after 14-day feature lag warm-up).
  - **Test Period:** 2015-06-02 to 2015-07-31 (60 days holdout).
- **Evaluation Metrics:**
  - **Mean Absolute Error (MAE):** Measures average magnitude of forecast errors in euros.
  - **Root Mean Squared Error (RMSE):** Penalizes larger forecasting deviations heavily.

---

## Model Performance

All models were evaluated on the exact same 60-day out-of-sample holdout test period:

| Rank | Model | Architecture / Order | MAE (€) | RMSE (€) |
| :---: | :--- | :--- | ---: | ---: |
| **1** | **LightGBM** | **Gradient Boosted Trees (Lags + Calendar)** | **323,834.04** | **451,188.96** |
| 2 | SARIMA | $(1,0,0) \times (1,0,1,7)$ | 1,122,513.00 | 1,540,606.00 |
| 3 | ARIMA | $(7,0,0)$ | 2,075,114.51 | 2,727,431.88 |
| 4 | Seasonal Naive | $y_{t} = y_{t-7}$ | 2,084,717.68 | 2,570,396.03 |
| 5 | Naive | $y_{t} = y_{t-1}$ | 2,746,028.78 | 4,278,452.33 |

### Key Forecast Highlights:
- **Best Classical Model:** **SARIMA(1,0,0)x(1,0,1,7)** effectively captured the weekly 7-day cyclicality, reducing MAE by ~46% over standard ARIMA.
- **Machine Learning Superiority:** **LightGBM** achieved the lowest forecasting error across both metrics, reducing **MAE by 71.15%** and **RMSE by 70.71%** compared to the best classical benchmark (SARIMA).
- **Feature Importance:** Autoregressive features (`Sales_Lag_1`, `Rolling_Mean_14`, `Sales_Lag_7`, `Rolling_Mean_7`) and calendar variables (`DayOfMonth`, `Month`, `DayOfWeek`) accounted for the majority of tree splits.

---

## Promotion Analysis

The promotion analysis evaluates open-store operating days (`Open == 1`, 844,392 observations):

### 1. Unadjusted Comparison (Open Stores)
- **Non-Promotion Days:** 467,496 observations (55.36%) | Mean Sales: **€5,929.41** (Median: €5,459)
- **Promotion Days:** 376,896 observations (44.64%) | Mean Sales: **€8,228.28** (Median: €7,649)
- **Unadjusted Difference:** **+€2,298.87 (+38.77%)**

### 2. Within-Store Consistency
- Out of **1,115** stores with both promotional and non-promotional periods, **1,114 stores (99.91%)** exhibited higher average sales during promotional periods.

### 3. Adjusted Regression (OLS)
An Ordinary Least Squares (OLS) model controlled for store fixed effects (`Store`), day of week (`DayOfWeek`), school holidays (`SchoolHoliday`), and state holidays (`StateHoliday`):
- **Promotion Coefficient ($\beta_{\text{Promo}}$):** **+€2,293.66**
- **Standard Error:** 3.63
- **95% Confidence Interval:** **[€2,286.54, €2,300.79]**
- **P-Value:** $< 0.001$

The estimated adjusted association (€2,293.66) is very close to the unadjusted open-store difference (€2,298.87), indicating that controlling for the included store and calendar factors did not materially alter the estimated association.

---

## Key Findings

- **Seasonality & Scheduling:** Strong 7-day cyclicality governs store demand; incorporating 7-day seasonal lags significantly outperforms standard autoregressive baselines.
- **Model Recommendation:** Gradient boosted trees (LightGBM) utilizing lag and rolling window features outperformed all classical time series models, delivering superior accuracy on turning points and holiday peaks.
- **Consistent Promotion Association:** Promotions showed a persistent and widespread positive relationship with sales across 99.91% of stores and across all weekdays.
- **Methodological Diligence:** Filtering for open-store trading days prevents zero-sales distortion, ensuring reliable comparisons.

---

## Limitations

- **Observational Data:** The promotion analysis is observational and does not prove direct causality. Unobserved factors (such as targeted marketing, localized foot traffic, or deliberate promotion scheduling during high-demand windows) may still influence sales.
- **Historical Stationarity:** Forecasting models assume future demand distributions reflect historical patterns and may require retraining under macroeconomic shifts or supply chain disruptions.

---

## Technologies Used

- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Statistical Modeling & Econometrics:** `statsmodels` (ARIMA, SARIMAX, OLS)
- **Machine Learning:** `lightgbm`, `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Interactive Computing:** `jupyter`

---

## Project Structure

```
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
```

---

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/retail-demand-forecasting.git
   cd retail-demand-forecasting
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Data Placement:**
   Ensure `train.csv`, `test.csv`, and `store.csv` are located inside `Data/raw/`.

5. **Run Notebooks in Sequence:**
   ```bash
   jupyter notebook
   ```
   Execute notebooks in numerical order:
   - `01_eda.ipynb`
   - `02_forecasting_classical.ipynb`
   - `03_forecasting_modern.ipynb`
   - `04_model_comparison.ipynb`
   - `05_promotion_analysis.ipynb`
