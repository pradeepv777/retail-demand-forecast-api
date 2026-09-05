"""
Smoke tests for the Rossmann Forecasting API.

These run against a live container started with SMOKE_TEST=true, so no real
model file or training CSV is needed. The goal is to confirm:
  - The container starts and the app is reachable
  - /health responds with the expected shape
  - /forecast returns 503 when no model is loaded (expected in smoke mode)
  - /forecast rejects invalid input with 422 (Pydantic validation — works
    regardless of whether a model is loaded)
  - /forecast/store returns 503 in smoke mode and rejects invalid input with 422
  - /forecast/store/whatif returns 503 in smoke mode and rejects invalid input with 422
"""

import os

import requests

BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


class TestHealth:
    def test_returns_200(self):
        r = requests.get(f"{BASE}/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_response_has_required_keys(self):
        r = requests.get(f"{BASE}/health")
        body = r.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "store_model_loaded" in body
        assert "last_known_date" in body

    def test_smoke_mode_model_not_loaded(self):
        """In SMOKE_TEST mode the model is intentionally not loaded."""
        r = requests.get(f"{BASE}/health")
        body = r.json()
        assert body["model_loaded"] is False

    def test_smoke_mode_store_model_not_loaded(self):
        """In SMOKE_TEST mode the store model is also intentionally not loaded."""
        r = requests.get(f"{BASE}/health")
        body = r.json()
        assert body["store_model_loaded"] is False

    def test_last_known_date_is_null_in_smoke_mode(self):
        r = requests.get(f"{BASE}/health")
        body = r.json()
        assert body["last_known_date"] is None


class TestForecastValidation:
    """These tests rely on Pydantic schema validation only — they pass
    regardless of whether the model is loaded, so they work in smoke mode."""

    def test_days_zero_is_rejected(self):
        r = requests.post(f"{BASE}/forecast", json={"days": 0})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_days_over_limit_is_rejected(self):
        r = requests.post(f"{BASE}/forecast", json={"days": 31})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_negative_days_is_rejected(self):
        r = requests.post(f"{BASE}/forecast", json={"days": -1})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_days_field_is_rejected(self):
        r = requests.post(f"{BASE}/forecast", json={})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_non_integer_days_is_rejected(self):
        r = requests.post(f"{BASE}/forecast", json={"days": "five"})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"


class TestForecastSmokeMode:
    """In SMOKE_TEST mode the model is None, so /forecast must return 503."""

    def test_valid_request_returns_503_when_no_model(self):
        r = requests.post(f"{BASE}/forecast", json={"days": 3})
        assert r.status_code == 503, f"Expected 503 in smoke mode, got {r.status_code}"

    def test_503_has_detail_message(self):
        r = requests.post(f"{BASE}/forecast", json={"days": 3})
        body = r.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# Per-store forecast — POST /forecast/store
# ---------------------------------------------------------------------------

class TestStoreForecastValidation:
    """Pydantic validation tests — pass in smoke mode (no model needed)."""

    def test_store_id_zero_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 0, "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_store_id_above_max_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1116, "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_store_id_negative_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": -1, "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_days_zero_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1, "days": 0})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_days_over_limit_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1, "days": 31})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_store_id_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_days_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_assume_promo_out_of_range_is_rejected(self):
        r = requests.post(
            f"{BASE}/forecast/store",
            json={"store_id": 1, "days": 7, "assume_promo": 2},
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_assume_school_holiday_out_of_range_is_rejected(self):
        r = requests.post(
            f"{BASE}/forecast/store",
            json={"store_id": 1, "days": 7, "assume_school_holiday": -1},
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_non_integer_store_id_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": "abc", "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"


class TestStoreForecastSmokeMode:
    """In SMOKE_TEST mode the store model is None, so /forecast/store must return 503."""

    def test_valid_request_returns_503_when_no_model(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1, "days": 3})
        assert r.status_code == 503, f"Expected 503 in smoke mode, got {r.status_code}"

    def test_503_has_detail_message(self):
        r = requests.post(f"{BASE}/forecast/store", json={"store_id": 1, "days": 3})
        body = r.json()
        assert "detail" in body


# ---------------------------------------------------------------------------
# What-if promo scenario — POST /forecast/store/whatif
# ---------------------------------------------------------------------------

class TestWhatIfValidation:
    """Pydantic validation tests — pass in smoke mode (no model needed)."""

    def test_store_id_zero_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 0, "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_store_id_above_max_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1116, "days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_days_zero_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1, "days": 0})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_days_over_limit_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1, "days": 31})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_store_id_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"days": 7})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_days_is_rejected(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"


class TestWhatIfSmokeMode:
    """In SMOKE_TEST mode the store model is None, so /forecast/store/whatif must return 503."""

    def test_valid_request_returns_503_when_no_model(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1, "days": 3})
        assert r.status_code == 503, f"Expected 503 in smoke mode, got {r.status_code}"

    def test_503_has_detail_message(self):
        r = requests.post(f"{BASE}/forecast/store/whatif", json={"store_id": 1, "days": 3})
        body = r.json()
        assert "detail" in body
