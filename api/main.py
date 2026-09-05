"""
Retail Demand Forecasting API

Exposes three forecast endpoints:

  GET  /health                   — liveness + model status
  POST /forecast                 — chain-wide daily sales (all 1,115 stores combined)
  POST /forecast/store           — per-store daily sales forecast
  POST /forecast/store/whatif    — promo vs no-promo comparison for a single store

Chain-wide model (lightgbm_model.pkl)
--------------------------------------
Trained on daily sales AGGREGATED ACROSS ALL STORES
(see notebooks/03_forecasting_modern.ipynb).

Per-store model (lightgbm_store_model.pkl)
-------------------------------------------
Trained on individual store-day rows with store metadata features
(see notebooks/06_store_level_forecasting.ipynb).
Features: Store, DayOfWeek, Month, Year, DayOfMonth, Promo, SchoolHoliday,
          StateHoliday, StoreType, Assortment, CompetitionDistance, Promo2,
          Sales_Lag_1, Sales_Lag_7, Sales_Lag_14,
          Sales_Rolling_Mean_7, Sales_Rolling_Mean_14

Recursive forecasting
---------------------
Because Sales_Lag_* and Sales_Rolling_Mean_* features require past sales
values that don't exist yet for future dates, forecasting more than one day
ahead is done RECURSIVELY: each day's prediction is fed back in as a lag
feature for the next day's prediction. Errors can compound over longer
horizons — this is disclosed in every response, not hidden.

SMOKE_TEST mode
---------------
SMOKE_TEST=true lets the app start without real model/data files.
/health returns model_loaded=false; /forecast and /forecast/store return 503.
Used by CI to verify the container starts and routes respond correctly.
"""

import contextlib
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DailyForecast,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    PromoScenario,
    StoreForecastRequest,
    StoreForecastResponse,
    WhatIfPromoRequest,
    WhatIfPromoResponse,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.pkl"
STORE_MODEL_PATH = BASE_DIR / "models" / "lightgbm_store_model.pkl"
STORE_FEATURES_PATH = BASE_DIR / "models" / "store_features.pkl"
TRAIN_CSV_PATH = BASE_DIR / "Data" / "raw" / "train.csv"

# ---------------------------------------------------------------------------
# Feature lists (must match training exactly)
# ---------------------------------------------------------------------------
CHAIN_FEATURES = [
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "StateHoliday_Count",
    "DayOfWeek_Num",
    "Month",
    "Year",
    "DayOfMonth",
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Lag_14",
    "Sales_Rolling_Mean_7",
    "Sales_Rolling_Mean_14",
]

STORE_FEATURES = [
    "Store",
    "DayOfWeek",
    "Month",
    "Year",
    "DayOfMonth",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Lag_14",
    "Sales_Rolling_Mean_7",
    "Sales_Rolling_Mean_14",
]

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

_state: Dict = {
    # Chain-wide model
    "model": None,
    "history": None,
    # Per-store model
    "store_model": None,
    "store_history": None,   # Dict[int, pd.DataFrame] keyed by store_id
    "store_features": None,  # Dict[int, dict] — encoded store metadata
}


# ---------------------------------------------------------------------------
# History builders (optimized for fast cold-start and low memory footprint)
# ---------------------------------------------------------------------------

def _build_daily_history(train_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized aggregation of chain-wide sales and holiday signals."""
    temp_df = train_df[["Date", "Sales", "Promo", "SchoolHoliday"]].copy()
    temp_df["StateHoliday_Count"] = (~train_df["StateHoliday"].isin([0, "0"])).astype(int)

    return (
        temp_df.groupby("Date")
        .agg(
            Sales=("Sales", "sum"),
            Promo=("Promo", "sum"),
            SchoolHoliday=("SchoolHoliday", "sum"),
            StateHoliday_Count=("StateHoliday_Count", "sum"),
        )
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _build_store_histories(train_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Build a lightweight per-store history dict keyed by store_id.

    Only Open==1 days and the essential inference columns (Date, Sales, Promo,
    SchoolHoliday) are retained. All lag and rolling calculations for future
    horizons are computed dynamically during recursive inference.
    """
    open_df = train_df.loc[
        train_df["Open"] == 1,
        ["Store", "Date", "Sales", "Promo", "SchoolHoliday"],
    ]
    return {
        int(store_id): group.reset_index(drop=True)
        for store_id, group in open_df.groupby("Store")
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app_: FastAPI):
    """Load models and build history on startup. Nothing to clean up on shutdown."""
    if os.getenv("SMOKE_TEST", "").lower() == "true":
        # All state remains None — /health reports not loaded, forecasts return 503
        yield
        return

    # --- Chain-wide model ---
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Chain model not found at {MODEL_PATH}")
    if not TRAIN_CSV_PATH.exists():
        raise RuntimeError(f"Training data not found at {TRAIN_CSV_PATH}")

    # Load raw training data once from disk and parse dates
    train_df = pd.read_csv(TRAIN_CSV_PATH, low_memory=False)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.sort_values("Date").reset_index(drop=True)

    _state["model"] = joblib.load(MODEL_PATH)
    _state["history"] = _build_daily_history(train_df)

    # --- Per-store model (optional — warn but don't crash if missing) ---
    if not STORE_MODEL_PATH.exists() or not STORE_FEATURES_PATH.exists():
        import warnings
        warnings.warn(
            "Per-store model or store_features.pkl not found. "
            "/forecast/store endpoints will return 503."
        )
        yield
        return

    _state["store_features"] = joblib.load(STORE_FEATURES_PATH)
    _state["store_model"] = joblib.load(STORE_MODEL_PATH)
    _state["store_history"] = _build_store_histories(train_df)
    yield



app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Serves chain-wide and per-store daily sales forecasts from LightGBM models "
        "trained on Rossmann Store Sales data."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    history = _state["history"]
    return HealthResponse(
        status="ok" if _state["model"] is not None else "model not loaded",
        model_loaded=_state["model"] is not None,
        store_model_loaded=_state["store_model"] is not None,
        last_known_date=history["Date"].max().date() if history is not None else None,
    )


# ---------------------------------------------------------------------------
# Helpers: store lookup and recursive forecasting
# ---------------------------------------------------------------------------

def _get_store_history(store_id: int) -> pd.DataFrame:
    """Validate store model status and return store history, or raise HTTP error."""
    if _state["store_model"] is None or _state["store_history"] is None:
        raise HTTPException(status_code=503, detail="Store model not loaded yet")
    if store_id not in _state["store_history"]:
        raise HTTPException(
            status_code=404,
            detail=f"Store {store_id} not found in training data.",
        )
    return _state["store_history"][store_id]


def _run_store_forecast(
    store_id: int,
    days: int,
    promo_val: float,
    school_val: float,
) -> tuple:
    """
    Returns (last_date, predictions_list).

    promo_val / school_val are the values to use for every forecast day.
    Caller is responsible for defaulting to trailing averages before calling.
    """
    store_hist = _state["store_history"][store_id]
    store_meta = _state["store_features"][store_id]
    model = _state["store_model"]

    sales_series = store_hist["Sales"].tolist()
    last_date = store_hist["Date"].max()

    predictions = []
    for step in range(1, days + 1):
        next_date = last_date + timedelta(days=step)

        row = {
            "Store": store_id,
            "DayOfWeek": next_date.dayofweek + 1,
            "Month": next_date.month,
            "Year": next_date.year,
            "DayOfMonth": next_date.day,
            "Promo": promo_val,
            "SchoolHoliday": school_val,
            "StateHoliday": 0,  # unknown for future; default to no state holiday
            "StoreType": store_meta["StoreType"],
            "Assortment": store_meta["Assortment"],
            "CompetitionDistance": store_meta["CompetitionDistance"],
            "Promo2": store_meta["Promo2"],
            "Sales_Lag_1": sales_series[-1],
            "Sales_Lag_7": sales_series[-7],
            "Sales_Lag_14": sales_series[-14],
            "Sales_Rolling_Mean_7": sum(sales_series[-7:]) / 7,
            "Sales_Rolling_Mean_14": sum(sales_series[-14:]) / 14,
        }

        X = pd.DataFrame([row])[STORE_FEATURES]
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # sales can't be negative

        predictions.append(
            DailyForecast(date=next_date.date(), predicted_sales=round(pred, 2))
        )
        sales_series.append(pred)

    return last_date, predictions


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """Forecast chain-wide total daily sales across all 1,115 Rossmann stores."""
    model = _state["model"]
    history = _state["history"]
    if model is None or history is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    recent = history.tail(7)
    promo_default = (
        request.assume_promo_active_stores
        if request.assume_promo_active_stores is not None
        else float(recent["Promo"].mean())
    )
    school_default = (
        request.assume_school_holiday_stores
        if request.assume_school_holiday_stores is not None
        else float(recent["SchoolHoliday"].mean())
    )
    state_default = (
        request.assume_state_holiday_stores
        if request.assume_state_holiday_stores is not None
        else float(recent["StateHoliday_Count"].mean())
    )

    sales_series = history["Sales"].tolist()
    last_date = history["Date"].max()

    predictions = []
    for step in range(1, request.days + 1):
        next_date = last_date + timedelta(days=step)

        row = {
            "DayOfWeek": next_date.dayofweek + 1,
            "Promo": promo_default,
            "SchoolHoliday": school_default,
            "StateHoliday_Count": state_default,
            "DayOfWeek_Num": next_date.dayofweek,
            "Month": next_date.month,
            "Year": next_date.year,
            "DayOfMonth": next_date.day,
            "Sales_Lag_1": sales_series[-1],
            "Sales_Lag_7": sales_series[-7],
            "Sales_Lag_14": sales_series[-14],
            "Sales_Rolling_Mean_7": sum(sales_series[-7:]) / 7,
            "Sales_Rolling_Mean_14": sum(sales_series[-14:]) / 14,
        }

        X = pd.DataFrame([row])[CHAIN_FEATURES]
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # sales can't be negative
        predictions.append(DailyForecast(date=next_date.date(), predicted_sales=round(pred, 2)))
        sales_series.append(pred)

    return ForecastResponse(
        last_known_date=last_date.date(),
        forecast_days=request.days,
        predictions=predictions,
        assumptions={
            "scope": "Chain-wide total daily sales across all stores, not a single store.",
            "promo_active_stores_assumed": round(promo_default, 2),
            "school_holiday_stores_assumed": round(school_default, 2),
            "state_holiday_stores_assumed": round(state_default, 2),
            "method": (
                "Recursive forecasting - later days use earlier predictions as lag inputs, "
                "so error can compound over the horizon."
            ),
        },
    )


@app.post("/forecast/store", response_model=StoreForecastResponse)
def forecast_store(request: StoreForecastRequest):
    """Forecast daily sales for a specific Rossmann store."""
    store_hist = _get_store_history(request.store_id)
    recent = store_hist.tail(7)

    promo_val = (
        float(request.assume_promo)
        if request.assume_promo is not None
        else float(recent["Promo"].mean())
    )
    school_val = (
        float(request.assume_school_holiday)
        if request.assume_school_holiday is not None
        else float(recent["SchoolHoliday"].mean())
    )

    last_date, predictions = _run_store_forecast(
        store_id=request.store_id,
        days=request.days,
        promo_val=promo_val,
        school_val=school_val,
    )

    return StoreForecastResponse(
        store_id=request.store_id,
        last_known_date=last_date.date(),
        forecast_days=request.days,
        predictions=predictions,
        assumptions={
            "scope": f"Per-store forecast for Store {request.store_id} only.",
            "promo_assumed": round(promo_val, 2),
            "school_holiday_assumed": round(school_val, 2),
            "state_holiday_assumed": 0,
            "method": (
                "Recursive forecasting - later days use earlier predictions as lag inputs, "
                "so error can compound over the horizon."
            ),
            "model_performance": "Test MAE ~659/store/day, RMSPE ~14.4% (held-out 2015-06 to 07)",
        },
    )


@app.post("/forecast/store/whatif", response_model=WhatIfPromoResponse)
def forecast_store_whatif(request: WhatIfPromoRequest):
    """
    Compare promo vs no-promo sales forecast for a specific store.

    Runs the same recursive forecast twice — once with Promo=1 every day,
    once with Promo=0 — and returns a day-by-day lift estimate.

    Historical A/B analysis (notebook 07) found a statistically significant
    promo lift of ~2,299/store/day (p < 0.0001, 95% CI [2,286, 2,312]).
    This endpoint lets you see what the model predicts for your specific store.
    """
    store_hist = _get_store_history(request.store_id)
    school_val = float(store_hist.tail(7)["SchoolHoliday"].mean())


    # Run both scenarios
    _, preds_with_promo = _run_store_forecast(
        store_id=request.store_id,
        days=request.days,
        promo_val=1.0,
        school_val=school_val,
    )
    _, preds_without_promo = _run_store_forecast(
        store_id=request.store_id,
        days=request.days,
        promo_val=0.0,
        school_val=school_val,
    )

    scenarios = []
    for with_p, without_p in zip(preds_with_promo, preds_without_promo):
        lift = round(with_p.predicted_sales - without_p.predicted_sales, 2)
        scenarios.append(
            PromoScenario(
                date=with_p.date,
                predicted_sales_with_promo=with_p.predicted_sales,
                predicted_sales_without_promo=without_p.predicted_sales,
                estimated_promo_lift=lift,
            )
        )

    total_lift = round(sum(s.estimated_promo_lift for s in scenarios), 2)

    return WhatIfPromoResponse(
        store_id=request.store_id,
        last_known_date=store_hist["Date"].max().date(),
        forecast_days=request.days,
        scenarios=scenarios,
        total_lift_over_period=total_lift,
        assumptions={
            "scope": f"What-if promo comparison for Store {request.store_id}.",
            "promo_on_scenario": "Promo=1 every forecast day",
            "promo_off_scenario": "Promo=0 every forecast day",
            "school_holiday_assumed": round(school_val, 2),
            "state_holiday_assumed": 0,
            "historical_reference": (
                "Notebook 07 A/B test found avg promo lift of ~2,299/store/day "
                "(p < 0.0001, 95% CI [2,286, 2,312]) across 844k store-day observations."
            ),
            "caution": (
                "This is a model-estimated effect, not a causal guarantee. "
                "Promotion assignment in the training data was not randomized."
            ),
            "method": (
                "Recursive forecasting - later days use earlier predictions as lag inputs, "
                "so error can compound over the horizon."
            ),
        },
    )
