"""Optional AI-assisted sales commentary using a server-side REST API request.

Only aggregate metrics are sent to the external model. The baseline forecasts and
error metrics are always calculated locally by forecasting.py.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests


def _currency(value: float) -> str:
    return f"₹{value:,.2f}"


def _local_summary(report: Any) -> str:
    """Reliable, API-free observations based only on computed project metrics."""
    metrics = report.metrics.sort_values("mae")
    best = metrics.iloc[0]
    daily = report.daily.sort_values("period")
    current = daily.tail(7)
    previous = daily.iloc[-14:-7] if len(daily) >= 14 else daily.iloc[:-7]
    if len(previous) and float(previous["revenue"].mean()) != 0:
        change = (float(current["revenue"].mean()) / float(previous["revenue"].mean()) - 1) * 100
        if change > 1:
            trend = f"Average daily revenue in the latest period is up {change:.1f}% versus the preceding period."
        elif change < -1:
            trend = f"Average daily revenue in the latest period is down {abs(change):.1f}% versus the preceding period."
        else:
            trend = "Average daily revenue is broadly stable compared with the preceding period."
    else:
        trend = "There is not enough earlier history to make a reliable period-over-period trend comparison."
    forecast_rows = report.forecast[report.forecast["method"] == best["method"]]
    total = float(forecast_rows["predicted"].sum())
    return "\n".join(
        [
            f"1. Trend: {trend}",
            f"2. Baseline check: {best['method']} performed best on the chronological holdout, with MAE {_currency(float(best['mae']))} and RMSE {_currency(float(best['rmse']))}.",
            f"3. Outlook: this method projects total revenue of {_currency(total)} over the next {report.horizon} days. Treat this as a baseline estimate, not a guarantee.",
            "4. Caution: these simple models do not include seasonality, promotions, stockouts, holidays, or other external factors. Use the error metrics and approximate forecast ranges when planning.",
        ]
    )


def _make_prompt(report: Any) -> str:
    daily = report.daily.sort_values("period")
    recent = daily.tail(7)
    prior = daily.iloc[-14:-7] if len(daily) >= 14 else daily.iloc[:-7]
    metrics = report.metrics.sort_values("mae")
    forecast_totals = (
        report.forecast.groupby("method", as_index=False)["predicted"].sum()
        .rename(columns={"predicted": "total_predicted_revenue"})
    )
    payload = {
        "period": {
            "first_date": str(daily["period"].min().date()),
            "last_date": str(daily["period"].max().date()),
            "daily_observations": int(len(daily)),
            "forecast_horizon_days": int(report.horizon),
        },
        "totals": {
            "revenue": round(float(daily["revenue"].sum()), 2),
            "units": int(daily["quantity"].sum()),
            "recent_7_day_revenue": round(float(recent["revenue"].sum()), 2),
            "previous_period_revenue": round(float(prior["revenue"].sum()), 2) if len(prior) else None,
        },
        "holdout_metrics": [
            {"method": str(row.method), "mae": round(float(row.mae), 2), "rmse": round(float(row.rmse), 2)}
            for row in metrics.itertuples()
        ],
        "forecast_total_by_method": [
            {"method": str(row.method), "total_predicted_revenue": round(float(row.total_predicted_revenue), 2)}
            for row in forecast_totals.itertuples()
        ],
        "limitations": list(report.limitations),
    }
    return (
        "Analyze this sales forecasting report using only the supplied data. "
        "Write 3 or 4 concise, plain-language numbered insights for a student capstone demo. "
        "Explain the trend, which baseline performed better on the chronological holdout, "
        "the forecast outlook, and one practical caution. Do not invent facts, causes, "
        "or precision; explicitly distinguish estimates from actuals. Mention that this "
        "is not a guarantee. Currency is INR.\n\n"
        f"Report data (JSON): {json.dumps(payload, ensure_ascii=False)}"
    )


def _extract_text(body: dict[str, Any]) -> str:
    """Extract output text from the Responses API JSON structure."""
    chunks: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"].strip())
    return "\n".join(chunks).strip()


def generate_ai_insights(report: Any, api_key: str | None = None, model: str | None = None) -> dict[str, str]:
    """Create API-based insights, falling back safely when no key/API is available."""
    fallback = _local_summary(report)
    api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return {
            "text": fallback,
            "source": "Local summary",
            "notice": "OPENAI_API_KEY is not configured, so these insights use local calculations. Add your own key in the .env file to enable the AI API.",
        }

    payload = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "instructions": (
            "You are a careful business analyst. Use only supplied aggregate metrics. "
            "Do not claim correlation or causation. Do not present baseline forecasts as certain. "
            "Use clear, brief language and INR currency."
        ),
        "input": _make_prompt(report),
        "max_output_tokens": 320,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        text = _extract_text(response.json())
        if not text:
            raise ValueError("The API response did not contain text output.")
        return {"text": text, "source": "OpenAI API", "notice": ""}
    except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
        return {
            "text": fallback,
            "source": "Local summary",
            "notice": "The AI API could not complete the request. Showing locally calculated insights instead; check your key, model, quota, and connection.",
        }



def _local_forecast_fallback(report: Any) -> dict[str, Any]:
    """Return the best measured baseline when an API is not configured or fails."""
    best_method = str(report.metrics.sort_values("mae").iloc[0]["method"])
    rows = report.forecast[report.forecast["method"] == best_method].sort_values("period")
    predictions = [
        {"date": pd.Timestamp(row.period).strftime("%Y-%m-%d"), "predicted_revenue": max(0.0, float(row.predicted))}
        for row in rows.itertuples()
    ]
    return {
        "source": "Local baseline fallback",
        "used_api": False,
        "summary": (
            f"The AI API is not active for this result. The local {best_method} baseline "
            f"is shown instead; its chronological holdout MAE is {_currency(float(report.metrics.set_index('method').loc[best_method, 'mae']))}."
        ),
        "insights": [
            f"The best baseline on the chronological holdout is {best_method}.",
            "The forecast is an estimate based on past sales, not a guaranteed outcome.",
            "Promotions, seasonality, holidays, stockouts, and external factors are not modeled by these simple baselines.",
        ],
        "forecast": predictions,
        "notice": "To enable AI-generated forecasts, set OPENAI_API_KEY in the project .env file. This result uses a measured local baseline, not AI-generated values.",
        "method_note": f"Fallback method: {best_method}",
    }


def generate_ai_forecast(report: Any, api_key: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Use the Responses API to analyze a daily sales series and return date-aligned AI forecasts.

    The model gets daily aggregated observations and baseline evaluation metrics, not raw transaction rows.
    Any invalid or incomplete API response safely falls back to the best local baseline.
    """
    import math

    fallback = _local_forecast_fallback(report)
    api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return fallback

    daily = report.daily.sort_values("period").tail(120)
    future_dates = sorted(pd.Timestamp(x).strftime("%Y-%m-%d") for x in report.forecast["period"].drop_duplicates())
    metrics = report.metrics.sort_values("mae")
    observations = [
        {
            "date": pd.Timestamp(row.period).strftime("%Y-%m-%d"),
            "revenue": round(float(row.revenue), 2),
            "units": int(row.quantity),
        }
        for row in daily.itertuples()
    ]
    baseline_metrics = [
        {"method": str(row.method), "mae": round(float(row.mae), 2), "rmse": round(float(row.rmse), 2)}
        for row in metrics.itertuples()
    ]
    prompt_data = {
        "currency": "INR",
        "recent_daily_observations": observations,
        "next_dates_to_forecast": future_dates,
        "horizon_days": report.horizon,
        "chronological_holdout_metrics": baseline_metrics,
        "baseline_forecast_by_method": [
            {
                "method": str(method),
                "predictions": [
                    {"date": pd.Timestamp(row.period).strftime("%Y-%m-%d"), "revenue": round(float(row.predicted), 2)}
                    for row in report.forecast[report.forecast["method"] == method].sort_values("period").itertuples()
                ],
            }
            for method in report.forecast["method"].drop_duplicates().tolist()
        ],
    }
    payload = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "instructions": (
            "You are a cautious sales forecasting analyst. Analyze the supplied daily time series, "
            "compare simple baselines, and generate a plausible forecast for each requested date. "
            "Do not invent known causes or external information. Use only the supplied aggregate data. "
            "Forecast values must be non-negative INR amounts. Explain uncertainty and data limitations. "
            "The AI forecast is advisory and not a guarantee. Output valid JSON only with exactly these keys: "
            "summary (string), insights (array of 3 short strings), method_note (string), limitations (string), "
            "forecast (array of objects with date and predicted_revenue). Return one forecast object for every "
            "requested date, preserving each date exactly."
        ),
        "input": "Analyze the series and produce an AI-assisted forecast. Data JSON:\n" + json.dumps(prompt_data, ensure_ascii=False),
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": min(1800, 350 + 55 * report.horizon),
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=35,
        )
        response.raise_for_status()
        raw = _extract_text(response.json())
        data = json.loads(raw)
        forecast_data = data.get("forecast")
        if not isinstance(forecast_data, list):
            raise ValueError("The API forecast was not a list.")
        parsed: dict[str, float] = {}
        for item in forecast_data:
            if not isinstance(item, dict):
                continue
            date = str(item.get("date", ""))[:10]
            value = float(item.get("predicted_revenue"))
            if date in future_dates and math.isfinite(value) and value >= 0:
                parsed[date] = value
        if set(parsed) != set(future_dates):
            raise ValueError("The AI response did not provide a valid value for every requested date.")
        insights = data.get("insights", [])
        if not isinstance(insights, list):
            insights = []
        insights = [str(item).strip() for item in insights if str(item).strip()][:4]
        return {
            "source": "OpenAI API",
            "used_api": True,
            "summary": str(data.get("summary", "AI-assisted forecast generated from aggregate daily sales." )).strip(),
            "insights": insights,
            "method_note": str(data.get("method_note", "AI-generated forecast informed by the supplied history and baseline metrics.")).strip(),
            "limitations_text": str(data.get("limitations", "Sales forecasts are estimates and may miss external factors.")).strip(),
            "forecast": [{"date": date, "predicted_revenue": round(parsed[date], 2)} for date in future_dates],
            "notice": "",
        }
    except (requests.RequestException, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        fallback["notice"] = (
            "The AI API could not return a valid forecast, so the app is showing the best local baseline instead. "
            "Check your API key, model access, quota, and internet connection."
        )
        return fallback
