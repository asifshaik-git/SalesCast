from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {"sale_date", "quantity", "unit_price"}


class ForecastError(ValueError):
    """Raised when uploaded or analytical data cannot be used safely."""


@dataclass
class ForecastReport:
    summary: pd.DataFrame
    daily: pd.DataFrame
    monthly: pd.DataFrame
    forecast: pd.DataFrame
    holdout: pd.DataFrame
    metrics: pd.DataFrame
    trend_chart: bytes
    forecast_chart: bytes
    horizon: int
    training_rows: int
    limitations: list[str]


def validate_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized sales columns while retaining optional analysis fields."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ForecastError(f"Missing required columns: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ForecastError("The uploaded table contains headers but no data rows.")

    clean = frame.copy()
    clean["sale_date"] = pd.to_datetime(clean["sale_date"], errors="coerce")
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce")
    clean["unit_price"] = pd.to_numeric(clean["unit_price"], errors="coerce")
    fractional_quantity = (
        clean["quantity"].notna()
        & np.isfinite(clean["quantity"])
        & (clean["quantity"] % 1 != 0)
    )
    if fractional_quantity.any():
        raise ForecastError(
            f"{int(fractional_quantity.sum())} record(s) have fractional quantities; "
            "quantity must be a positive whole number."
        )
    invalid = (
        clean["sale_date"].isna()
        | clean["quantity"].isna()
        | clean["unit_price"].isna()
        | ~np.isfinite(clean["quantity"])
        | ~np.isfinite(clean["unit_price"])
        | (clean["quantity"] <= 0)
        | (clean["unit_price"] < 0)
    )
    if invalid.any():
        raise ForecastError(
            f"{int(invalid.sum())} record(s) have missing or invalid dates, quantities, or prices."
        )
    clean["sale_date"] = clean["sale_date"].dt.strftime("%Y-%m-%d")
    clean["quantity"] = clean["quantity"].astype(int)
    clean["unit_price"] = clean["unit_price"].astype(float)

    for column, default in (("region", "Unknown"), ("product_name", "Imported Product"), ("category", "Imported")):
        if column in clean.columns:
            clean[column] = clean[column].fillna(default).astype(str).str.strip()
            clean.loc[clean[column] == "", column] = default
        else:
            clean[column] = default
    clean["revenue"] = clean["quantity"] * clean["unit_price"]
    return clean[["sale_date", "quantity", "unit_price", "region", "revenue", "product_name", "category"]]


_HEADER_ALIASES = {
    "sale_date": {"saledate", "salesdate", "date", "orderdate", "transactiondate", "invoicedate", "purchasedate", "dateofsale", "createdat", "timestamp", "datetime", "day"},
    "quantity": {"quantity", "qty", "units", "unitssold", "quantitysold", "numbersold", "soldquantity", "itemquantity", "itemcount", "numberofitems", "volume"},
    "unit_price": {"unitprice", "priceperunit", "sellingprice", "saleprice", "itemprice", "unitcostprice", "rate", "unitcost", "productprice"},
    "sales_amount": {"sales", "totalsales", "revenue", "amount", "linetotal", "totalamount", "ordertotal", "salesamount", "totalrevenue", "netsales", "extendedprice"},
    "region": {"region", "salesregion", "area", "territory", "location", "state", "city", "country", "market", "zone"},
    "product_name": {"productname", "product", "producttitle", "item", "itemname", "sku", "skuname", "productid", "itemid"},
    "category": {"category", "productcategory", "itemcategory", "producttype", "type", "segment", "department"},
}


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def detect_column_mapping(columns: list[object]) -> dict[str, str | None]:
    """Map common sales-column names to the project's standard fields."""
    by_normalized = {_normalise_header(column): str(column) for column in columns}
    found: dict[str, str | None] = {}
    for field, aliases in _HEADER_ALIASES.items():
        found[field] = next((by_normalized[a] for a in aliases if a in by_normalized), None)
    # A generic "sales"/"revenue" field represents a line total unless a distinct
    # price-per-unit column was found. Never interpret total sales as unit price.
    found["price_column"] = found.get("unit_price") or found.get("sales_amount")
    found["price_mode"] = "unit_price" if found.get("unit_price") else "sales_amount"
    return found


def read_uploaded_table(filename: str, raw: bytes) -> pd.DataFrame:
    """Read CSV, TSV, TXT, DAT, XLSX, or XLSM files and infer headers/delimiters."""
    suffix = Path(filename or "").suffix.lower()
    if not raw:
        raise ForecastError("The selected file is empty.")
    try:
        if suffix in {".xlsx", ".xlsm"}:
            frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        elif suffix in {".csv", ".tsv", ".txt", ".dat"}:
            if suffix == ".tsv":
                frame = pd.read_csv(io.BytesIO(raw), sep="\t", encoding="utf-8-sig")
            else:
                frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        else:
            raise ForecastError("Use CSV, TSV, TXT, DAT, XLSX, or XLSM files.")
    except ForecastError:
        raise
    except (ValueError, OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ImportError) as exc:
        if suffix == ".xls":
            raise ForecastError("Older .xls workbooks are not supported. Save the workbook as .xlsx and upload it again.") from exc
        raise ForecastError(f"The file could not be read as a table: {exc}") from exc

    # If no recognizable headers exist and the *parsed header values themselves*
    # look like the first data row, retry without a header. This does not mistake
    # custom text headers (for example, "Transaction Day") for headerless data.
    recognized = detect_column_mapping([str(c).strip() for c in frame.columns])
    if not any(recognized.get(field) for field in ("sale_date", "quantity", "unit_price", "sales_amount")):
        try:
            if suffix not in {".xlsx", ".xlsm"} and len(frame.columns) >= 3:
                header_date = pd.to_datetime(pd.Series([str(frame.columns[0])]), errors="coerce").notna().iloc[0]
                header_quantity = pd.to_numeric(pd.Series([str(frame.columns[1])]), errors="coerce").notna().iloc[0]
                header_amount = pd.to_numeric(pd.Series([str(frame.columns[2])]), errors="coerce").notna().iloc[0]
                if header_date and header_quantity and header_amount:
                    if suffix == ".tsv":
                        headerless = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, encoding="utf-8-sig")
                    else:
                        headerless = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None, encoding="utf-8-sig")
                    headerless.columns = [f"Column {i + 1}" for i in range(headerless.shape[1])]
                    frame = headerless
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            pass

    if frame.empty or not len(frame.columns):
        raise ForecastError("The file has no data rows or columns.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        frame.columns = [column or f"Column {index + 1}" for index, column in enumerate(frame.columns)]
    if len(set(frame.columns)) != len(frame.columns):
        frame.columns = [f"{column} ({index + 1})" for index, column in enumerate(frame.columns)]
    return frame


def normalize_uploaded_frame(frame: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Apply detected or user-selected source-column mappings."""
    for field in ("sale_date", "quantity", "price_column"):
        column = mapping.get(field)
        if not column or column not in frame.columns:
            raise ForecastError("Select a date column, quantity column, and price/total-sales column.")
    date_column = str(mapping["sale_date"])
    quantity_column = str(mapping["quantity"])
    price_column = str(mapping["price_column"])
    output = pd.DataFrame({
        "sale_date": frame[date_column],
        "quantity": frame[quantity_column],
    })
    price_values = pd.to_numeric(frame[price_column], errors="coerce")
    if mapping.get("price_mode") == "sales_amount":
        quantity_values = pd.to_numeric(frame[quantity_column], errors="coerce")
        output["unit_price"] = price_values / quantity_values.where(quantity_values != 0)
    else:
        output["unit_price"] = price_values
    for field in ("region", "product_name", "category"):
        column = mapping.get(field)
        if column and column in frame.columns:
            output[field] = frame[column]
    return output


def import_sales_frame(database: str | Path, frame: pd.DataFrame) -> int:
    clean = validate_sales_frame(frame).drop_duplicates(
        subset=["sale_date", "quantity", "unit_price", "region", "product_name"], keep="first"
    )
    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_database_schema(connection)
        imported = 0
        for row in clean.itertuples(index=False):
            product_name = str(row.product_name or "Imported Product")
            category = str(row.category or "Imported")
            region = str(row.region or "Unknown")
            connection.execute(
                "INSERT OR IGNORE INTO products(product_name, category) VALUES (?, ?)",
                (product_name, category),
            )
            product_id = connection.execute(
                "SELECT product_id FROM products WHERE product_name = ?", (product_name,)
            ).fetchone()[0]
            key = _record_key(row.sale_date, row.quantity, row.unit_price, region, product_name)
            already_exists = connection.execute(
                "SELECT 1 FROM sales WHERE record_key = ? OR "
                "(product_id = ? AND sale_date = ? AND quantity = ? AND unit_price = ? AND COALESCE(region, 'Unknown') = ?)",
                (key, product_id, row.sale_date, int(row.quantity), float(row.unit_price), region),
            ).fetchone()
            if already_exists:
                continue
            connection.execute(
                """INSERT INTO sales(
                       product_id, sale_date, quantity, unit_price, region, record_key
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (product_id, row.sale_date, int(row.quantity), float(row.unit_price), region, key),
            )
            imported += 1
    return imported

def _record_key(
    sale_date: str, quantity: float, unit_price: float,
    region: str = "Unknown", product_name: str = "Imported Product",
) -> str:
    value = f"{sale_date}|{quantity:.12g}|{unit_price:.12g}|{product_name}|{region}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ensure_database_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK(quantity > 0),
            unit_price REAL NOT NULL CHECK(unit_price >= 0),
            region TEXT,
            record_key TEXT,
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        );
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            method TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sales)").fetchall()}
    if "record_key" not in columns:
        connection.execute("ALTER TABLE sales ADD COLUMN record_key TEXT")


def import_sales_csv(database: str | Path, stream: BinaryIO) -> int:
    """Backward-compatible CSV importer used by existing callers and tests."""
    try:
        frame = pd.read_csv(stream)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ForecastError("The uploaded file is not a readable CSV.") from exc
    return import_sales_frame(database, frame)


def load_sales_detail(database: str | Path) -> pd.DataFrame:
    """Return imported rows for the user-configured analysis page."""
    with sqlite3.connect(database) as connection:
        detail = pd.read_sql_query(
            """SELECT s.sale_date, s.quantity, s.unit_price,
                      COALESCE(s.region, 'Unknown') AS region,
                      p.product_name, p.category,
                      (s.quantity * s.unit_price) AS revenue
               FROM sales s JOIN products p ON p.product_id = s.product_id
               ORDER BY s.sale_date, s.sale_id""",
            connection,
            parse_dates=["sale_date"],
        )
    return detail


def build_analysis_chart(
    detail: pd.DataFrame,
    dimension: str = "date",
    chart_type: str = "line",
    measure: str = "revenue",
    frequency: str = "daily",
) -> bytes:
    """Build a chart based on the analysis controls selected by the user."""
    allowed_dimensions = {"date", "region", "product", "category"}
    allowed_types = {"line", "bar", "area", "scatter", "pie"}
    allowed_measures = {"revenue", "quantity"}
    if dimension not in allowed_dimensions or chart_type not in allowed_types or measure not in allowed_measures:
        raise ForecastError("Choose a supported graph, dimension, and measure.")
    if frequency not in {"daily", "monthly"}:
        raise ForecastError("Choose a daily or monthly interval.")
    if detail.empty:
        raise ForecastError("Upload data before creating a graph.")
    if chart_type == "pie" and dimension == "date":
        raise ForecastError("Pie charts need a category such as region, product, or category.")

    data = detail.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce")
    value_column = "revenue" if measure == "revenue" else "quantity"
    if dimension == "date":
        period = "D" if frequency == "daily" else "MS"
        grouped = data.dropna(subset=["sale_date"]).set_index("sale_date")[value_column].resample(period).sum()
        if frequency == "daily":
            grouped = grouped.asfreq("D", fill_value=0)
        x_values = grouped.index
        y_values = grouped.values
        labels = None
        title_dimension = f"{frequency.title()} sales"
    else:
        column = {"region": "region", "product": "product_name", "category": "category"}[dimension]
        if dimension == "region" and not (data["region"].fillna("Unknown").astype(str) != "Unknown").any():
            raise ForecastError("No region column was detected in the uploaded data.")
        if dimension == "product" and not (data["product_name"].fillna("Imported Product").astype(str) != "Imported Product").any():
            raise ForecastError("No product-name column was detected in the uploaded data.")
        if dimension == "category" and not (data["category"].fillna("Imported").astype(str) != "Imported").any():
            raise ForecastError("No category column was detected in the uploaded data.")
        grouped = data.groupby(column, dropna=False)[value_column].sum().sort_values(ascending=False).head(15)
        labels = [str(value) if pd.notna(value) else "Unknown" for value in grouped.index]
        x_values = np.arange(len(grouped))
        y_values = grouped.values
        title_dimension = {"region": "Region", "product": "Product", "category": "Category"}[dimension] + " comparison"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if chart_type == "pie":
        if not len(y_values) or float(np.sum(y_values)) <= 0:
            plt.close(fig)
            raise ForecastError("A pie chart needs positive values to display.")
        ax.pie(y_values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title_dimension + " · " + measure.title())
    elif chart_type == "bar":
        ax.bar(x_values, y_values, color="#2563eb")
        ax.set_title(title_dimension + " · " + measure.title())
        if labels is not None:
            ax.set_xticks(x_values, labels, rotation=30, ha="right")
    elif chart_type == "area":
        ax.fill_between(x_values, y_values, alpha=0.22, color="#2563eb")
        ax.plot(x_values, y_values, color="#2563eb")
        ax.set_title(title_dimension + " · " + measure.title())
        if labels is not None:
            ax.set_xticks(x_values, labels, rotation=30, ha="right")
    elif chart_type == "scatter":
        ax.scatter(x_values, y_values, color="#2563eb", alpha=0.8)
        ax.set_title(title_dimension + " · " + measure.title())
        if labels is not None:
            ax.set_xticks(x_values, labels, rotation=30, ha="right")
    else:
        ax.plot(x_values, y_values, marker="o", color="#2563eb", linewidth=2)
        ax.set_title(title_dimension + " · " + measure.title())
        if labels is not None:
            ax.set_xticks(x_values, labels, rotation=30, ha="right")
    if chart_type != "pie":
        ax.set_ylabel("Revenue" if measure == "revenue" else "Units sold")
        ax.grid(axis="y", alpha=0.22)
    if dimension == "date" and len(x_values):
        fig.autofmt_xdate()
    fig.tight_layout()
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


def load_sales(database: str | io.PathLike[str]) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(
            """SELECT sale_date, SUM(quantity * unit_price) AS revenue,
                      SUM(quantity) AS quantity, COUNT(*) AS transaction_count
               FROM sales GROUP BY sale_date ORDER BY sale_date""",
            connection,
            parse_dates=["sale_date"],
        )


def aggregate_sales(sales: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["period", "revenue", "quantity"])
    data = sales.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    if "revenue" not in data:
        data["revenue"] = data["quantity"] * data["unit_price"]
    grouped = (
        data.set_index("sale_date")[["revenue", "quantity"]]
        .resample(frequency)
        .sum()
    )
    if frequency == "D":
        grouped = grouped.asfreq("D", fill_value=0)
    return grouped.reset_index(names="period")


def _series(sales: pd.DataFrame) -> pd.Series:
    daily = aggregate_sales(sales)
    if daily.empty:
        raise ForecastError("No sales data is available.")
    return daily.set_index("period")["revenue"]


def _moving_average(values: np.ndarray, window: int = 7) -> float:
    return float(np.mean(values[-min(window, len(values)) :]))


def _linear_regression(values: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(values), dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return np.repeat(float(values[-1]), horizon)
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + horizon, dtype=float)
    return np.maximum(0, np.polyval([slope, intercept], future_x))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute forecast error on the chronological holdout."""
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error; larger misses receive a higher penalty."""
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _interval(predictions: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    margin = 1.96 * spread
    return np.maximum(0, predictions - margin), predictions + margin


def _chart_bytes(
    series: pd.Series, forecast: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[bytes, bytes]:
    trend = io.BytesIO()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(series.index, series.values, color="#2563eb", label="Daily revenue")
    axes[0].set_title("Daily sales trend")
    axes[0].set_ylabel("Revenue (INR)")
    axes[0].grid(alpha=0.25)
    for method in holdout["method"].unique():
        rows = holdout[holdout["method"] == method]
        axes[1].plot(rows["period"], rows["actual"], color="#172033", marker="o", label="Actual")
        axes[1].plot(rows["period"], rows["predicted"], marker="o", label=f"{method} predicted")
    axes[1].set_title("Chronological holdout: actual versus predicted")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Revenue (INR)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(trend, format="png", dpi=140)
    plt.close(fig)

    future = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 4))
    recent = series.tail(45)
    ax.plot(recent.index, recent.values, color="#2563eb", label="Actual revenue")
    for method in forecast["method"].unique():
        rows = forecast[forecast["method"] == method]
        ax.plot(rows["period"], rows["predicted"], marker="o", label=method)
        ax.fill_between(
            rows["period"],
            rows["lower"],
            rows["upper"],
            alpha=0.12,
            label=f"{method} approximate interval",
        )
    ax.set_title("Future sales forecast with approximate uncertainty")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (INR)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(future, format="png", dpi=140)
    plt.close(fig)
    return trend.getvalue(), future.getvalue()


def build_forecast_report(sales: pd.DataFrame, horizon: int = 7) -> ForecastReport:
    series = _series(sales)
    if len(series) < 6:
        raise ForecastError("At least 6 daily observations are required.")
    horizon = max(1, min(int(horizon), 30))
    holdout_size = max(1, min(horizon, len(series) // 4))
    train, test = series.iloc[:-holdout_size], series.iloc[-holdout_size:]
    methods = {
        "7-day moving average": np.repeat(_moving_average(train.values), len(test)),
        "linear regression": _linear_regression(train.values, len(test)),
    }
    holdout_parts = []
    metric_rows = []
    for method, predictions in methods.items():
        holdout_parts.append(
            pd.DataFrame(
                {
                    "period": test.index,
                    "method": method,
                    "actual": test.values,
                    "predicted": predictions,
                }
            )
        )
        metric_rows.append({
            "method": method,
            "mae": _mae(test.values, predictions),
            "rmse": _rmse(test.values, predictions),
        })
    holdout = pd.concat(holdout_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("mae")

    future_dates = [series.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
    future_methods = {
        "7-day moving average": np.repeat(_moving_average(series.values), horizon),
        "linear regression": _linear_regression(series.values, horizon),
    }
    forecast_parts = []
    for method, predictions in future_methods.items():
        residuals = methods[method] - test.values
        lower, upper = _interval(predictions, residuals)
        forecast_parts.append(
            pd.DataFrame(
                {
                    "period": future_dates,
                    "method": method,
                    "predicted": predictions,
                    "predicted_value": predictions,
                    "lower": lower,
                    "upper": upper,
                }
            )
        )
    forecast = pd.concat(forecast_parts, ignore_index=True)
    daily = aggregate_sales(sales)
    monthly = aggregate_sales(sales, "MS")
    summary = pd.DataFrame(
        [
            {
                "label": "Transactions",
                "value": int(sales.get("transaction_count", pd.Series([len(sales)])).sum()),
            },
            {"label": "Daily observations", "value": int(len(daily))},
            {"label": "Total units", "value": float(daily["quantity"].sum())},
            {"label": "Total revenue", "value": float(daily["revenue"].sum())},
            {"label": "Average daily revenue", "value": float(daily["revenue"].mean())},
        ]
    )
    trend_chart, forecast_chart = _chart_bytes(series, forecast, holdout)
    limitations = [
        "Intervals use 1.96 times the sample residual standard deviation and are approximate, especially with small datasets.",
        f"MAE is measured on the chronological final {holdout_size}-day holdout; no random shuffling is used.",
        "The baselines do not model promotions, seasonality, holidays, stockouts, product mix, or external events.",
    ]
    return ForecastReport(
        summary=summary,
        daily=daily,
        monthly=monthly,
        forecast=forecast,
        holdout=holdout,
        metrics=metrics,
        trend_chart=trend_chart,
        forecast_chart=forecast_chart,
        horizon=horizon,
        training_rows=len(train),
        limitations=limitations,
    )
