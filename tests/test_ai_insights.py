import io

import pandas as pd

from ai_insights import generate_ai_forecast, generate_ai_insights
from forecasting import build_forecast_report


def _report():
    frame = pd.DataFrame(
        {
            "sale_date": pd.date_range("2026-01-01", periods=20),
            "quantity": list(range(1, 21)),
            "unit_price": [10.0] * 20,
        }
    )
    return build_forecast_report(frame, horizon=5)


def test_ai_insights_works_without_api_key():
    result = generate_ai_insights(_report(), api_key="")
    assert result["source"] == "Local summary"
    assert "MAE" in result["text"]
    assert "RMSE" in result["text"]
    assert result["notice"]


def test_ai_insights_uses_responses_api(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "1. Trend: revenue increased."}
                        ],
                    }
                ]
            }

    calls = {}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("ai_insights.requests.post", fake_post)
    result = generate_ai_insights(_report(), api_key="test-key", model="test-model")
    assert result["source"] == "OpenAI API"
    assert result["text"] == "1. Trend: revenue increased."
    assert calls["url"].endswith("/v1/responses")
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["json"]["model"] == "test-model"


def test_ai_api_failure_falls_back(monkeypatch):
    def fake_post(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("ai_insights.requests.post", fake_post)
    result = generate_ai_insights(_report(), api_key="test-key")
    assert result["source"] == "Local summary"
    assert "could not complete" in result["notice"]


def test_ai_forecast_uses_api_and_returns_date_aligned_predictions(monkeypatch):
    report = _report()
    dates = sorted(pd.to_datetime(report.forecast["period"]).drop_duplicates().dt.strftime("%Y-%m-%d").tolist())

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            import json
            body = {
                "summary": "Recent daily revenue varies, so plan cautiously.",
                "insights": ["The series has uneven daily sales.", "Compare both baseline errors.", "Review forecasts against actuals."],
                "method_note": "AI forecast based on supplied daily observations.",
                "limitations": "Small histories may be unreliable.",
                "forecast": [{"date": date, "predicted_revenue": 123.45 + i} for i, date in enumerate(dates)],
            }
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(body)}]}]}

    calls = {}
    def fake_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("ai_insights.requests.post", fake_post)
    result = generate_ai_forecast(report, api_key="test-key", model="test-model")
    assert result["source"] == "OpenAI API"
    assert result["used_api"] is True
    assert [item["date"] for item in result["forecast"]] == dates
    assert calls["url"].endswith("/v1/responses")
    assert calls["json"]["text"]["format"]["type"] == "json_object"


def test_ai_forecast_without_key_uses_baseline():
    result = generate_ai_forecast(_report(), api_key="")
    assert result["source"] == "Local baseline fallback"
    assert result["used_api"] is False
    assert len(result["forecast"]) == 5
