import io
import sqlite3

import pandas as pd
import pytest

from forecasting import (
    ForecastError,
    aggregate_sales,
    build_forecast_report,
    import_sales_csv,
    validate_sales_frame,
)


def sample_frame():
    dates = pd.date_range("2026-01-01", periods=12)
    return pd.DataFrame({"sale_date": dates, "quantity": range(1, 13), "unit_price": 10})


def test_validation_calculates_revenue():
    clean = validate_sales_frame(sample_frame())
    assert clean["revenue"].iloc[-1] == 120
    assert clean["sale_date"].iloc[0] == "2026-01-01"


def test_validation_rejects_invalid_values():
    frame = sample_frame()
    frame.loc[0, "quantity"] = 0
    with pytest.raises(ForecastError, match="invalid"):
        validate_sales_frame(frame)


def test_validation_rejects_fractional_quantity():
    frame = sample_frame()
    frame["quantity"] = frame["quantity"].astype(float)
    frame.loc[0, "quantity"] = 1.5
    with pytest.raises(ForecastError, match="whole number"):
        validate_sales_frame(frame)


def test_report_uses_chronological_holdout_and_two_methods():
    report = build_forecast_report(sample_frame(), horizon=4)
    assert set(report.metrics["method"]) == {"7-day moving average", "linear regression"}
    assert len(report.forecast) == 8
    assert report.training_rows == 9
    assert report.forecast_chart.startswith(b"\x89PNG")


def test_import_requires_required_columns():
    with pytest.raises(ForecastError, match="Missing required"):
        validate_sales_frame(pd.DataFrame({"date": ["2026-01-01"]}))


def test_import_rejects_empty_csv(tmp_path):
    with pytest.raises(ForecastError, match="readable CSV"):
        import_sales_csv(tmp_path / "nested" / "sales.db", io.BytesIO(b""))


def test_import_creates_nested_database_directory(tmp_path):
    database = tmp_path / "nested" / "sales.db"
    csv = io.BytesIO(
        b"sale_date,quantity,unit_price\n2026-01-01,1,2\n"
    )
    assert import_sales_csv(database, csv) == 1


def test_daily_and_monthly_aggregation():
    frame = sample_frame()
    daily = aggregate_sales(frame)
    monthly = aggregate_sales(frame, "MS")
    assert len(daily) == 12
    assert daily["revenue"].sum() == 780
    assert len(monthly) == 1
    assert monthly["quantity"].iloc[0] == 78


def test_import_skips_duplicate_records(tmp_path):
    database = tmp_path / "sales.db"
    csv = io.BytesIO(
        b"sale_date,quantity,unit_price\n"
        b"2026-01-01,10,20\n"
        b"2026-01-01,10,20\n"
    )
    assert import_sales_csv(database, csv) == 1
    assert import_sales_csv(database, io.BytesIO(
        b"sale_date,quantity,unit_price\n2026-01-01,10,20\n"
    )) == 0


def test_duplicate_import_preserves_existing_data(tmp_path):
    database = tmp_path / "sales.db"
    csv = b"sale_date,quantity,unit_price\n2026-01-01,10,20\n"
    assert import_sales_csv(database, io.BytesIO(csv)) == 1
    assert import_sales_csv(database, io.BytesIO(csv)) == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 1
        assert connection.execute(
            "SELECT quantity, unit_price FROM sales"
        ).fetchone() == (10, 20.0)


def test_forecast_contains_holdout_and_uncertainty_columns():
    report = build_forecast_report(sample_frame(), horizon=4)
    assert {"actual", "predicted"}.issubset(report.holdout.columns)
    assert {"predicted", "lower", "upper"}.issubset(report.forecast.columns)
    assert (report.forecast["lower"] <= report.forecast["predicted"]).all()
    assert (report.forecast["predicted"] <= report.forecast["upper"]).all()


def test_constant_series_is_safe():
    frame = sample_frame()
    frame["quantity"] = 10
    report = build_forecast_report(frame)
    assert (report.forecast["predicted"] == 100).all()


def test_optional_region_is_preserved():
    frame = sample_frame().assign(region="South")
    clean = validate_sales_frame(frame)
    assert set(clean["region"]) == {"South"}


def test_auto_column_detection_and_sales_total_conversion():
    from forecasting import detect_column_mapping, normalize_uploaded_frame, read_uploaded_table
    raw = (
        b"ORDER_DATE,QTY,SALES,REGION,PRODUCT_NAME\n"
        b"2026-01-01,2,40,North,Widget\n"
        b"2026-01-02,1,15,South,Gizmo\n"
    )
    frame = read_uploaded_table("sales.csv", raw)
    mapping = detect_column_mapping(list(frame.columns))
    assert mapping["sale_date"] == "ORDER_DATE"
    assert mapping["quantity"] == "QTY"
    assert mapping["price_column"] == "SALES"
    normalized = normalize_uploaded_frame(frame, mapping)
    assert normalized["unit_price"].tolist() == [20.0, 15.0]
    assert normalized["product_name"].tolist() == ["Widget", "Gizmo"]


def test_headerless_delimited_file_gets_detectable_columns():
    from forecasting import read_uploaded_table
    frame = read_uploaded_table("sales.csv", b"2026-01-01,2,40\n2026-01-02,1,15\n")
    assert list(frame.columns) == ["Column 1", "Column 2", "Column 3"]
    assert len(frame) == 2


def test_analysis_chart_supports_user_selected_graphs(tmp_path):
    from forecasting import build_analysis_chart, import_sales_frame, load_sales_detail
    frame = sample_frame().assign(region=["North", "South"] * 6, product_name=["A", "B"] * 6)
    database = tmp_path / "analysis.db"
    import_sales_frame(database, frame)
    detail = load_sales_detail(database)
    assert build_analysis_chart(detail, dimension="date", chart_type="line").startswith(b"\x89PNG")
    assert build_analysis_chart(detail, dimension="region", chart_type="pie").startswith(b"\x89PNG")
