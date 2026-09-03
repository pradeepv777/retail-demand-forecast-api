from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chain-wide forecast (existing)
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    days: int = Field(..., ge=1, le=30, description="Number of future days to forecast (1-30)")
    assume_promo_active_stores: Optional[float] = Field(
        None,
        description=(
            "Optional override: number of stores assumed to be running a promo on each "
            "forecasted day. If omitted, the average from the last 7 known days is used."
        ),
    )
    assume_school_holiday_stores: Optional[float] = Field(
        None,
        description="Optional override for the SchoolHoliday feature. Defaults to the last-7-day average.",
    )
    assume_state_holiday_stores: Optional[float] = Field(
        None,
        description="Optional override for the StateHoliday_Count feature. Defaults to the last-7-day average.",
    )


class DailyForecast(BaseModel):
    date: date
    predicted_sales: float


class ForecastResponse(BaseModel):
    last_known_date: date
    forecast_days: int
    predictions: List[DailyForecast]
    assumptions: dict = Field(
        ...,
        description=(
            "This model forecasts TOTAL daily sales summed across all 1,115 Rossmann stores, "
            "not any single store. Promo/holiday features for future dates are estimates, not "
            "known facts, since they haven't happened yet."
        ),
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    store_model_loaded: bool = False
    last_known_date: Optional[date] = None


# ---------------------------------------------------------------------------
# Per-store forecast  —  POST /forecast/store
# ---------------------------------------------------------------------------

class StoreForecastRequest(BaseModel):
    store_id: int = Field(..., ge=1, le=1115, description="Rossmann store ID (1–1115)")
    days: int = Field(..., ge=1, le=30, description="Number of future days to forecast (1-30)")
    assume_promo: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Override whether this store runs a promo on each forecasted day (0 or 1). "
            "If omitted, defaults to the store's trailing 7-day average."
        ),
    )
    assume_school_holiday: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Override SchoolHoliday for each forecasted day (0 or 1). Defaults to trailing 7-day average.",
    )


class StoreForecastResponse(BaseModel):
    store_id: int
    last_known_date: date
    forecast_days: int
    predictions: List[DailyForecast]
    assumptions: dict = Field(
        ...,
        description=(
            "Per-store forecast using a LightGBM model trained on individual store-day rows. "
            "Promo and SchoolHoliday for future dates are estimates unless overridden by the caller."
        ),
    )


# ---------------------------------------------------------------------------
# What-if promo scenario  —  POST /forecast/store/whatif
# ---------------------------------------------------------------------------

class WhatIfPromoRequest(BaseModel):
    store_id: int = Field(..., ge=1, le=1115, description="Rossmann store ID (1–1115)")
    days: int = Field(..., ge=1, le=30, description="Number of future days to compare (1-30)")


class PromoScenario(BaseModel):
    date: date
    predicted_sales_with_promo: float
    predicted_sales_without_promo: float
    estimated_promo_lift: float = Field(
        ...,
        description="predicted_sales_with_promo minus predicted_sales_without_promo",
    )


class WhatIfPromoResponse(BaseModel):
    store_id: int
    last_known_date: date
    forecast_days: int
    scenarios: List[PromoScenario]
    total_lift_over_period: float = Field(
        ...,
        description="Sum of estimated_promo_lift across all forecast days.",
    )
    assumptions: dict = Field(
        ...,
        description=(
            "Two forecasts are run for the same store: one with Promo=1 every day, "
            "one with Promo=0. The lift is the model's estimated marginal effect of "
            "running a promotion. Historical A/B analysis (notebook 06) found an "
            "average per-store promo lift of ~2,299/day across all store types."
        ),
    )
