import io

import pandas as pd
import pytest
from app import create_app


def sample_csv(rows=12):
    lines = ["sale_date,quantity,unit_price,region,product_name,category"]
    for day in range(1, rows + 1):
        lines.append(f"2026-01-{day:02d},{day},20,North,Widget,Hardware")
    return "\n".join(lines).encode()


def test_dashboard_starts_empty(tmp_path):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Your insights are one upload away" in response.data
    assert b".xlsx" in response.data
    assert b"Upload" in response.data


def test_csv_import_forecast_and_analysis_controls(tmp_path, monkeypatch):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    response = client.post(
        "/import",
        data={"file": (io.BytesIO(sample_csv()), "sales.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Detected 12 rows and 6 columns" in response.data
    assert b"Imported 12 valid sales records" in response.data

    analysis = client.get("/analysis")
    assert analysis.status_code == 200
    assert b"Your sales" in analysis.data
    assert b"Imported rows" in analysis.data
    assert b"INTERACTIVE GRAPH" in analysis.data
    assert b"update the page automatically" in analysis.data
    assert b"Graph type" in analysis.data
    assert b"Daily sales table" in analysis.data
    assert b"Monthly sales table" in analysis.data

    selected = client.get(
        "/analysis?show=1&show_chart=1&summary=1&graph_type=bar&dimension=date&measure=revenue&frequency=daily&table=monthly"
    )
    assert selected.status_code == 200
    assert b"INTERACTIVE GRAPH" in selected.data
    assert b"Monthly sales" in selected.data
    image = client.get("/analysis/chart.png?dimension=date&graph_type=bar&measure=revenue&frequency=daily")
    assert image.status_code == 200
    assert image.mimetype == "image/png"

    forecast_page = client.get("/forecast")
    assert forecast_page.status_code == 200
    assert b"AI analysis and predictions" in forecast_page.data
    assert b"holdout-chart" in forecast_page.data
    assert b"future-chart" in forecast_page.data
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ai_result = client.get("/api/ai-forecast?horizon=5")
    assert ai_result.status_code == 200
    assert ai_result.json["source"] == "Local baseline fallback"
    assert len(ai_result.json["forecast"]) == 5
    assert client.get("/chart/trend").mimetype == "image/png"
    report_download = client.get("/report.csv")
    assert report_download.status_code == 200
    assert report_download.mimetype == "text/csv"
    assert b"period,method,predicted" in report_download.data


def test_auto_maps_sales_total_to_unit_price(tmp_path):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    csv = b"DATE,QUANTITY,SALES\n2026-01-01,2,40\n2026-01-02,4,100\n"
    response = client.post(
        "/import",
        data={"file": (io.BytesIO(csv), "sales.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Imported 2 valid sales records" in response.data
    from forecasting import load_sales_detail
    data = load_sales_detail(app.config["DATABASE"])
    assert data["unit_price"].tolist() == [20.0, 25.0]


def test_uncertain_columns_show_mapping_screen(tmp_path):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    csv = b"Transaction Day,Number Sold,Line Revenue\n2026-01-01,2,40\n2026-01-02,1,15\n"
    response = client.post(
        "/import",
        data={"file": (io.BytesIO(csv), "custom.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Confirm your" in response.data
    assert b"rows and 3 columns" in response.data
    assert b"Validate and import" in response.data


def test_excel_upload(tmp_path):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    buffer = io.BytesIO()
    pd.DataFrame({
        "sale_date": pd.date_range("2026-01-01", periods=6),
        "quantity": [1, 2, 3, 4, 5, 6],
        "unit_price": [10, 10, 10, 10, 10, 10],
    }).to_excel(buffer, index=False, engine="openpyxl")
    response = client.post(
        "/import",
        data={"file": (io.BytesIO(buffer.getvalue()), "sales.xlsx")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Imported 6 valid sales records" in response.data


def test_reset_clears_imported_data(tmp_path):
    app = create_app(tmp_path / "test.db")
    client = app.test_client()
    client.post(
        "/import",
        data={"file": (io.BytesIO(sample_csv(6)), "sales.csv")},
        content_type="multipart/form-data",
    )
    response = client.post("/reset", follow_redirects=True)
    assert response.status_code == 200
    assert b"All imported sales data and forecasts were reset" in response.data
    assert b"Your insights are one upload away" in response.data


def test_production_requires_secret_key(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app(tmp_path / "production.db")


def test_configured_secret_key_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(tmp_path / "production.db")
    assert app.config["SECRET_KEY"] == "test-secret"
