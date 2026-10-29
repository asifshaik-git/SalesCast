from __future__ import annotations

import io
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from ai_insights import generate_ai_forecast, generate_ai_insights
from forecasting import (
    ForecastError,
    build_analysis_chart,
    build_forecast_report,
    detect_column_mapping,
    import_sales_frame,
    load_sales,
    load_sales_detail,
    normalize_uploaded_frame,
    read_uploaded_table,
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get("SALES_FORECAST_DB", BASE_DIR / "sales_forecast.db"))
load_dotenv(BASE_DIR / ".env")
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xlsm"}


def _cache_upload(app: Flask, frame: pd.DataFrame) -> str:
    cache_dir = Path(app.config["UPLOAD_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Remove abandoned mapping previews after four hours.
    cutoff = time.time() - 4 * 60 * 60
    for item in cache_dir.glob("*.pkl"):
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
        except OSError:
            pass
    token = secrets.token_urlsafe(24)
    frame.to_pickle(cache_dir / f"{token}.pkl")
    return token


def _cached_upload_path(app: Flask, token: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,80}", token or ""):
        raise ForecastError("This column-mapping session is invalid. Please upload the file again.")
    path = Path(app.config["UPLOAD_CACHE_DIR"]) / f"{token}.pkl"
    if not path.is_file():
        raise ForecastError("This column-mapping session expired. Please upload the file again.")
    return path


def _available_mapping(mapping: dict[str, str | None]) -> bool:
    return bool(mapping.get("sale_date") and mapping.get("quantity") and mapping.get("price_column"))


def _save_upload_summary(
    filename: str, frame: pd.DataFrame, mapping: dict[str, str | None], imported: int
) -> None:
    session["last_upload"] = {
        "filename": str(filename)[:150],
        "source_rows": int(len(frame)),
        "source_columns": int(len(frame.columns)),
        "imported_rows": int(imported),
        "mapping": {
            key: value for key, value in mapping.items()
            if key in {"sale_date", "quantity", "price_column", "price_mode", "region", "product_name", "category"}
        },
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _build_analysis_tables(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if detail.empty:
        return {key: pd.DataFrame() for key in ("daily", "monthly", "region", "product", "category")}
    data = detail.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce")
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce").fillna(0)
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)

    daily_source = data.assign(period=data["sale_date"].dt.floor("D"))
    daily = daily_source.groupby("period", as_index=False).agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("sale_date", "size")
    )

    monthly_source = data.assign(period=data["sale_date"].dt.to_period("M").dt.to_timestamp())
    monthly = monthly_source.groupby("period", as_index=False).agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("sale_date", "size")
    )

    def grouped(column: str, label: str) -> pd.DataFrame:
        result = data.groupby(column, dropna=False, as_index=False).agg(
            revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("sale_date", "size")
        )
        result = result.rename(columns={column: label})
        return result.sort_values("revenue", ascending=False).head(30)

    return {
        "daily": daily.tail(90).sort_values("period", ascending=False),
        "monthly": monthly.tail(36).sort_values("period", ascending=False),
        "region": grouped("region", "region"),
        "product": grouped("product_name", "product_name"),
        "category": grouped("category", "category"),
    }


def _analysis_plot_spec(
    detail: pd.DataFrame,
    dimension: str = "date",
    chart_type: str = "line",
    measure: str = "revenue",
    frequency: str = "daily",
) -> dict:
    """Build a browser-interactive Plotly spec for the current analysis controls."""
    from forecasting import ForecastError

    if detail.empty:
        raise ForecastError("Upload data before creating a graph.")
    if dimension not in {"date", "region", "product", "category"}:
        dimension = "date"
    if chart_type not in {"line", "bar", "area", "scatter", "pie"}:
        chart_type = "line"
    if measure not in {"revenue", "quantity"}:
        measure = "revenue"
    if frequency not in {"daily", "monthly"}:
        frequency = "daily"
    if chart_type == "pie" and dimension == "date":
        chart_type = "line"

    value_label = "Revenue" if measure == "revenue" else "Units sold"
    hover_value = "₹%{y:,.2f}" if measure == "revenue" else "%{y:,.0f} units"
    data = detail.copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce")
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce").fillna(0)
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)

    if dimension == "date":
        if frequency == "monthly":
            data["period"] = data["sale_date"].dt.to_period("M").dt.to_timestamp()
            grouped = data.groupby("period", as_index=False).agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        else:
            data["period"] = data["sale_date"].dt.floor("D")
            grouped = data.groupby("period", as_index=False).agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
            if not grouped.empty:
                all_dates = pd.date_range(grouped["period"].min(), grouped["period"].max(), freq="D")
                grouped = grouped.set_index("period").reindex(all_dates, fill_value=0).rename_axis("period").reset_index()
        grouped = grouped.sort_values("period")
        x_values = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in grouped["period"]]
        y_values = [float(value) for value in grouped[measure]]
        if chart_type == "pie":
            chart_type = "line"
        trace_type = "bar" if chart_type == "bar" else "scatter"
        trace = {"type": trace_type, "x": x_values, "y": y_values, "name": value_label}
        if chart_type == "bar":
            trace.update({"hovertemplate": "%{x}<br>" + value_label + ": " + hover_value + "<extra></extra>"})
        else:
            trace.update({"mode": "markers" if chart_type == "scatter" else "lines+markers"})
            if chart_type == "area":
                trace["fill"] = "tozeroy"
                trace["fillcolor"] = "rgba(37,99,235,0.14)"
            trace["hovertemplate"] = "%{x}<br>" + value_label + ": " + hover_value + "<extra></extra>"
        layout = {
            "height": 460,
            "margin": {"l": 68, "r": 24, "t": 20, "b": 46},
            "hovermode": "x unified",
            "showlegend": False,
            "xaxis": {"title": "", "showticklabels": False, "showspikes": True, "spikemode": "across", "spikedash": "dot", "spikethickness": 1},
            "yaxis": {"title": value_label, "rangemode": "tozero", "gridcolor": "#e5e7eb"},
            "uirevision": f"analysis-{dimension}-{chart_type}-{measure}-{frequency}",
        }
        return {"data": [trace], "layout": layout, "config": {"responsive": True, "displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}}

    column = {"region": "region", "product": "product_name", "category": "category"}[dimension]
    data[column] = data[column].fillna("Unknown").astype(str).replace("", "Unknown")
    grouped = data.groupby(column, as_index=False, dropna=False).agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum"), transactions=("sale_date", "size")
    ).sort_values(measure, ascending=False).head(12)
    labels = [str(value) for value in grouped[column]]
    values = [float(value) for value in grouped[measure]]
    if chart_type == "pie":
        trace = {
            "type": "pie", "labels": labels, "values": values, "hole": 0.28,
            "textinfo": "label+percent", "hovertemplate": "%{label}<br>" + value_label + ": %{value:,.2f}<extra></extra>"
            if measure == "revenue" else "%{label}<br>Units sold: %{value:,.0f}<extra></extra>",
        }
        layout = {"height": 460, "margin": {"l": 20, "r": 20, "t": 16, "b": 16}, "showlegend": True, "legend": {"orientation": "h", "y": -0.05}, "uirevision": f"analysis-{dimension}-pie-{measure}"}
    else:
        trace_type = "bar" if chart_type == "bar" else "scatter"
        trace = {"type": trace_type, "x": labels, "y": values, "name": value_label}
        if chart_type == "bar":
            trace["hovertemplate"] = "%{x}<br>" + value_label + ": " + hover_value + "<extra></extra>"
        else:
            trace["mode"] = "markers" if chart_type == "scatter" else "lines+markers"
            if chart_type == "area":
                trace["fill"] = "tozeroy"
                trace["fillcolor"] = "rgba(37,99,235,0.14)"
            trace["hovertemplate"] = "%{x}<br>" + value_label + ": " + hover_value + "<extra></extra>"
        layout = {
            "height": 460, "margin": {"l": 68, "r": 24, "t": 20, "b": 90}, "hovermode": "closest",
            "showlegend": False, "xaxis": {"title": column.replace("_", " ").title(), "tickangle": -25},
            "yaxis": {"title": value_label, "rangemode": "tozero", "gridcolor": "#e5e7eb"},
            "uirevision": f"analysis-{dimension}-{chart_type}-{measure}",
        }
    return {"data": [trace], "layout": layout, "config": {"responsive": True, "displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}}


def _forecast_plot_specs(report) -> tuple[dict, dict]:
    """Interactive holdout and future-forecast charts; dates appear on hover only."""
    holdout = report.holdout.copy()
    holdout["period"] = pd.to_datetime(holdout["period"])
    actual = holdout.drop_duplicates("period").sort_values("period")
    holdout_x = [value.strftime("%Y-%m-%d") for value in actual["period"]]
    holdout_traces = [{
        "type": "scatter", "mode": "lines+markers", "name": "Actual",
        "x": holdout_x, "y": [float(value) for value in actual["actual"]],
        "line": {"width": 3, "color": "#172033"}, "hovertemplate": "%{x}<br>Actual: ₹%{y:,.2f}<extra></extra>",
    }]
    colors = ["#2563eb", "#f97316", "#16a34a", "#8b5cf6"]
    for index, method in enumerate(holdout["method"].drop_duplicates().tolist()):
        rows = holdout[holdout["method"] == method].sort_values("period")
        holdout_traces.append({
            "type": "scatter", "mode": "lines+markers", "name": str(method),
            "x": [value.strftime("%Y-%m-%d") for value in rows["period"]],
            "y": [float(value) for value in rows["predicted"]],
            "line": {"width": 2, "color": colors[index % len(colors)], "dash": "dot"},
            "hovertemplate": "%{x}<br>" + str(method) + ": ₹%{y:,.2f}<extra></extra>",
        })
    shared_layout = {
        "height": 460, "margin": {"l": 76, "r": 26, "t": 72, "b": 42},
        "hovermode": "x unified", "legend": {"orientation": "h", "y": 1.08, "yanchor": "bottom", "x": 0, "xanchor": "left"},
        "xaxis": {"title": "", "showticklabels": False, "showspikes": True, "spikemode": "across", "spikedash": "dot", "spikethickness": 1},
        "yaxis": {"title": "Revenue (INR)", "rangemode": "tozero", "gridcolor": "#e5e7eb"},
        "uirevision": f"forecast-{report.horizon}",
    }
    holdout_spec = {"data": holdout_traces, "layout": dict(shared_layout), "config": {"responsive": True, "displaylogo": False, "scrollZoom": True}}

    future = report.forecast.copy()
    future["period"] = pd.to_datetime(future["period"])
    best_method = str(report.metrics.sort_values("mae").iloc[0]["method"])
    best_rows = future[future["method"] == best_method].sort_values("period")
    future_x = [value.strftime("%Y-%m-%d") for value in best_rows["period"]]
    forecast_traces = [
        {"type": "scatter", "mode": "lines", "name": "Approximate range lower", "x": future_x,
         "y": [float(value) for value in best_rows["lower"]], "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "name": "Approximate range", "x": future_x,
         "y": [float(value) for value in best_rows["upper"]], "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(37,99,235,0.15)", "showlegend": False, "hoverinfo": "skip"},
    ]
    for index, method in enumerate(future["method"].drop_duplicates().tolist()):
        rows = future[future["method"] == method].sort_values("period")
        forecast_traces.append({
            "type": "scatter", "mode": "lines+markers", "name": str(method),
            "x": [value.strftime("%Y-%m-%d") for value in rows["period"]],
            "y": [float(value) for value in rows["predicted"]],
            "line": {"width": 2.5, "color": colors[index % len(colors)]},
            "hovertemplate": "%{x}<br>" + str(method) + ": ₹%{y:,.2f}<extra></extra>",
        })
    forecast_spec = {"data": forecast_traces, "layout": dict(shared_layout), "config": {"responsive": True, "displaylogo": False, "scrollZoom": True}}
    return holdout_spec, forecast_spec


def create_app(database: Path | str = DATABASE) -> Flask:
    app = Flask(__name__)
    environment = os.environ.get("APP_ENV", "development").lower()
    configured_secret = os.environ.get("SECRET_KEY")
    if environment == "production" and not configured_secret:
        raise RuntimeError("SECRET_KEY must be set when APP_ENV=production.")
    app.config.update(
        DATABASE=Path(database),
        SECRET_KEY=configured_secret or secrets.token_hex(32),
        APP_ENV=environment,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        UPLOAD_CACHE_DIR=BASE_DIR / ".upload_cache",
        AI_CACHE_DIR=BASE_DIR / ".ai_cache",
    )
    Path(app.config["UPLOAD_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["AI_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    with app.app_context():
        init_db(app.config["DATABASE"])

    @app.get("/")
    def dashboard():
        detail = load_sales_detail(app.config["DATABASE"])
        last_upload = session.get("last_upload")
        return render_template(
            "dashboard.html",
            detail_rows=len(detail),
            last_upload=last_upload,
            has_forecast_data=len(load_sales(app.config["DATABASE"])) >= 6,
        )

    @app.post("/import")
    def import_data():
        session.pop("ai_insights", None)
        session.pop("ai_forecast_cache", None)
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            flash("Choose a data file to upload.", "error")
            return redirect(url_for("dashboard"))
        filename = upload.filename.replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            flash("Use CSV, TSV, TXT, DAT, XLSX, or XLSM files.", "error")
            return redirect(url_for("dashboard"))
        try:
            frame = read_uploaded_table(filename, upload.read())
            mapping = detect_column_mapping(list(frame.columns))
            if _available_mapping(mapping):
                normalized = normalize_uploaded_frame(frame, mapping)
                imported = import_sales_frame(app.config["DATABASE"], normalized)
                _save_upload_summary(filename, frame, mapping, imported)
                flash(
                    f"Detected {len(frame):,} rows and {len(frame.columns):,} columns. "
                    f"Imported {imported:,} valid sales records; duplicates were skipped.",
                    "success",
                )
                return redirect(url_for("analysis"))
            token = _cache_upload(app, frame)
            preview_rows = frame.head(6).fillna("").astype(str).to_dict(orient="records")
            return render_template(
                "mapping.html",
                token=token,
                filename=filename,
                columns=list(frame.columns),
                mapping=mapping,
                row_count=len(frame),
                column_count=len(frame.columns),
                preview_rows=preview_rows,
            )
        except ForecastError as exc:
            flash(str(exc), "error")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            flash(f"Could not process this file: {exc}", "error")
        return redirect(url_for("dashboard"))

    @app.post("/map-columns")
    def map_columns():
        token = request.form.get("token", "")
        path: Path | None = None
        try:
            path = _cached_upload_path(app, token)
            frame = pd.read_pickle(path)
            mapping: dict[str, str | None] = {
                "sale_date": request.form.get("sale_date") or None,
                "quantity": request.form.get("quantity") or None,
                "price_column": request.form.get("price_column") or None,
                "price_mode": request.form.get("price_mode", "unit_price"),
                "region": request.form.get("region") or None,
                "product_name": request.form.get("product_name") or None,
                "category": request.form.get("category") or None,
            }
            normalized = normalize_uploaded_frame(frame, mapping)
            imported = import_sales_frame(app.config["DATABASE"], normalized)
            filename = Path(request.form.get("filename", "uploaded-file")).name
            _save_upload_summary(filename, frame, mapping, imported)
            session.pop("ai_insights", None)
            session.pop("ai_forecast_cache", None)
            flash(
                f"Detected {len(frame):,} rows and {len(frame.columns):,} columns. "
                f"Imported {imported:,} valid sales records; duplicates were skipped.",
                "success",
            )
            return redirect(url_for("analysis"))
        except ForecastError as exc:
            flash(str(exc), "error")
            return redirect(url_for("dashboard"))
        except (OSError, ValueError, KeyError) as exc:
            flash(f"Could not apply the column mapping: {exc}", "error")
            return redirect(url_for("dashboard"))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    @app.post("/reset")
    def reset_data():
        session.pop("ai_insights", None)
        session.pop("ai_forecast_cache", None)
        session.pop("last_upload", None)
        with sqlite3.connect(app.config["DATABASE"]) as connection:
            connection.execute("DELETE FROM forecasts")
            connection.execute("DELETE FROM sales")
            connection.execute("DELETE FROM products")
        flash("All imported sales data and forecasts were reset.", "success")
        return redirect(url_for("dashboard"))

    @app.get("/analysis")
    def analysis():
        detail = load_sales_detail(app.config["DATABASE"])
        tables = _build_analysis_tables(detail)
        initialized = request.args.get("initialized") == "1"
        selected_tables = set(request.args.getlist("table"))
        show_summary = (not initialized) or request.args.get("summary") == "1"
        show_chart = (not initialized) or request.args.get("show_chart") == "1"
        has_regions = not detail.empty and bool((detail["region"].fillna("Unknown").astype(str) != "Unknown").any())
        has_products = not detail.empty and bool((detail["product_name"].fillna("Imported Product").astype(str) != "Imported Product").any())
        has_categories = not detail.empty and bool((detail["category"].fillna("Imported").astype(str) != "Imported").any())
        dimension = request.args.get("dimension", "date")
        if dimension not in {"date", "region", "product", "category"}:
            dimension = "date"
        if dimension == "region" and not has_regions:
            dimension = "date"
        if dimension == "product" and not has_products:
            dimension = "date"
        if dimension == "category" and not has_categories:
            dimension = "date"
        graph_type = request.args.get("graph_type", "line")
        if graph_type not in {"line", "bar", "area", "scatter", "pie"}:
            graph_type = "line"
        if graph_type == "pie" and dimension == "date":
            graph_type = "line"
        measure = request.args.get("measure", "revenue")
        if measure not in {"revenue", "quantity"}:
            measure = "revenue"
        frequency = request.args.get("frequency", "daily")
        if frequency not in {"daily", "monthly"}:
            frequency = "daily"
        available_rows = len(detail)
        summary = {
            "rows": available_rows,
            "total_revenue": float(detail["revenue"].sum()) if available_rows else 0.0,
            "total_units": int(detail["quantity"].sum()) if available_rows else 0,
            "date_min": str(detail["sale_date"].min().date()) if available_rows else "—",
            "date_max": str(detail["sale_date"].max().date()) if available_rows else "—",
        }
        chart_spec = None
        if show_chart and not detail.empty:
            try:
                chart_spec = _analysis_plot_spec(detail, dimension, graph_type, measure, frequency)
            except ForecastError:
                chart_spec = None
        return render_template(
            "analysis.html",
            detail=detail,
            tables=tables,
            show_results=True,
            selected_tables=selected_tables,
            show_summary=show_summary,
            show_chart=show_chart,
            dimension=dimension,
            graph_type=graph_type,
            measure=measure,
            frequency=frequency,
            has_regions=has_regions,
            has_products=has_products,
            has_categories=has_categories,
            summary=summary,
            chart_spec=chart_spec,
            last_upload=session.get("last_upload"),
        )

    @app.get("/analysis/chart.png")
    def analysis_chart():
        detail = load_sales_detail(app.config["DATABASE"])
        try:
            image = build_analysis_chart(
                detail,
                dimension=request.args.get("dimension", "date"),
                chart_type=request.args.get("graph_type", "line"),
                measure=request.args.get("measure", "revenue"),
                frequency=request.args.get("frequency", "daily"),
            )
        except ForecastError as exc:
            return (str(exc), 400)
        return send_file(io.BytesIO(image), mimetype="image/png", download_name="sales-analysis.png")

    @app.get("/forecast")
    def forecast():
        sales = load_sales(app.config["DATABASE"])
        if len(sales) < 6:
            flash("Import at least 6 dated observations before opening the forecast tab.", "error")
            return redirect(url_for("dashboard"))
        try:
            horizon = int(request.args.get("horizon", 7))
        except ValueError:
            horizon = 7
        horizon = max(1, min(horizon, 30))
        try:
            report = build_forecast_report(sales, horizon=horizon)
            holdout_plot, future_plot = _forecast_plot_specs(report)
        except ForecastError as exc:
            flash(str(exc), "error")
            return redirect(url_for("dashboard"))
        return render_template(
            "forecast.html",
            report=report,
            holdout_plot=holdout_plot,
            future_plot=future_plot,
            ai_api_configured=bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        )

    @app.get("/api/ai-forecast")
    def ai_forecast_api():
        sales = load_sales(app.config["DATABASE"])
        if len(sales) < 6:
            return jsonify({"error": "Import at least 6 dated observations before generating a forecast."}), 400
        try:
            horizon = int(request.args.get("horizon", 7))
        except (TypeError, ValueError):
            horizon = 7
        horizon = max(1, min(horizon, 30))
        report = build_forecast_report(sales, horizon=horizon)
        signature_source = sales.to_json(orient="records", date_format="iso")
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12] if api_key else "no-key"
        signature = hashlib.sha256(f"{signature_source}|{horizon}|{model}|{key_fingerprint}".encode("utf-8")).hexdigest()
        cache_file = Path(app.config["AI_CACHE_DIR"]) / f"{signature}.json"
        result = None
        if cache_file.is_file():
            try:
                result = json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                result = None
        if not isinstance(result, dict):
            result = generate_ai_forecast(report, api_key=api_key, model=model)
            # Cache only successful API results; local fallbacks should retry when configuration changes.
            if result.get("used_api"):
                try:
                    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
        return jsonify(result)

    @app.post("/ai-insights")
    def ai_insights():
        sales = load_sales(app.config["DATABASE"])
        if len(sales) < 6:
            flash("Import at least 6 dated observations before generating insights.", "error")
            return redirect(url_for("dashboard"))
        try:
            horizon = int(request.form.get("horizon", 7))
        except (TypeError, ValueError):
            horizon = 7
        horizon = max(1, min(horizon, 30))
        report = build_forecast_report(sales, horizon=horizon)
        result = generate_ai_insights(
            report,
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        )
        session["ai_insights"] = result
        if result.get("notice"):
            flash(result["notice"], "info")
        return redirect(url_for("forecast", horizon=horizon))

    @app.get("/chart/<chart_name>")
    def chart(chart_name: str):
        sales = load_sales(app.config["DATABASE"])
        if len(sales) < 6:
            return ("Not enough data to create a chart.", 400)
        try:
            horizon = int(request.args.get("horizon", 7))
        except (TypeError, ValueError):
            horizon = 7
        horizon = max(1, min(horizon, 30))
        report = build_forecast_report(sales, horizon=horizon)
        charts = {"forecast": report.forecast_chart, "trend": report.trend_chart}
        if chart_name not in charts:
            return ("Chart not found.", 404)
        return send_file(
            io.BytesIO(charts[chart_name]),
            mimetype="image/png",
            download_name=f"{chart_name}.png",
        )

    @app.get("/report.csv")
    def download_report():
        sales = load_sales(app.config["DATABASE"])
        if len(sales) < 6:
            flash("At least 6 dated observations are required for a forecast report.", "error")
            return redirect(url_for("dashboard"))
        try:
            horizon = int(request.args.get("horizon", 7))
        except (TypeError, ValueError):
            horizon = 7
        report = build_forecast_report(sales, horizon=max(1, min(horizon, 30)))
        output = io.StringIO()
        metric_lookup = report.metrics.set_index("method")["mae"].to_dict()
        rmse_lookup = report.metrics.set_index("method")["rmse"].to_dict()
        forecast_export = report.forecast.copy()
        forecast_export["actual"] = None
        forecast_export["absolute_error"] = None
        forecast_export["record_type"] = "forecast"
        forecast_export["mae"] = forecast_export["method"].map(metric_lookup)
        forecast_export["rmse"] = forecast_export["method"].map(rmse_lookup)
        holdout_export = report.holdout.copy()
        holdout_export["lower"] = None
        holdout_export["upper"] = None
        holdout_export["absolute_error"] = (holdout_export["actual"] - holdout_export["predicted"]).abs()
        holdout_export["record_type"] = "holdout"
        holdout_export["mae"] = holdout_export["method"].map(metric_lookup)
        holdout_export["rmse"] = holdout_export["method"].map(rmse_lookup)
        columns = ["period", "method", "predicted", "lower", "upper", "actual", "absolute_error", "mae", "rmse", "record_type"]
        pd.concat([forecast_export.reindex(columns=columns), holdout_export.reindex(columns=columns)], ignore_index=True).to_csv(output, index=False)
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"salescast-report-{datetime.now():%Y%m%d}.csv",
        )

    return app


def init_db(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
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
                FOREIGN KEY(product_id) REFERENCES products(product_id)
            );
            CREATE TABLE IF NOT EXISTS forecasts (
                forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT NOT NULL,
                method TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sales)").fetchall()}
        if "record_key" not in columns:
            connection.execute("ALTER TABLE sales ADD COLUMN record_key TEXT")


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
