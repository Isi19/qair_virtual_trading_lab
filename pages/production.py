"""Production forecast accuracy for the selected delivery period."""

import altair as alt
import streamlit as st

from src.dashboard.components import chart_card
from src.dashboard.data import (
    filter_period,
    load_asset_forecast,
    load_table,
    market_zone_config,
    selected_market_zone_id,
)


zone_id = selected_market_zone_id()
zone_config = market_zone_config()
period_start, period_end = st.session_state["delivery_period"]

st.title("Production Forecast")

# Choose one asset or the portfolio for the page's accuracy metrics.
asset_options = ["Portfolio"] + [asset["name"] for asset in zone_config["assets"]]
asset_column, day_column, _ = st.columns([1.2, 1.2, 2.6])
with asset_column:
    selected_asset_name = st.selectbox(
        "Asset",
        options=asset_options,
        key=f"production_asset_{zone_id}",
    )

if selected_asset_name == "Portfolio":
    forecast = load_table("portfolio")
    actual_column = "portfolio_actual_mw"
    p10_column = "portfolio_p10_mw"
    p50_column = "portfolio_p50_mw"
    p90_column = "portfolio_p90_mw"
    installed_capacity_mw = sum(
        asset["capacity_mw"] for asset in zone_config["assets"]
    )
else:
    selected_asset = next(
        asset for asset in zone_config["assets"]
        if asset["name"] == selected_asset_name
    )
    forecast = load_asset_forecast(selected_asset["id"])
    actual_column = "actual_generation_mw"
    p10_column = "generation_p10_mw"
    p50_column = "generation_p50_mw"
    p90_column = "generation_p90_mw"
    installed_capacity_mw = selected_asset["capacity_mw"]

period_forecast = filter_period(forecast, period_start, period_end)
delivery_days = sorted(period_forecast["delivery_start_local"].dt.date.unique())

if not delivery_days:
    st.info("No production forecasts are available for the selected period.")
    st.stop()

# The selected day will control the intraday chart added in the next step.
with day_column:
    selected_day = st.selectbox(
        "Delivery day",
        options=delivery_days,
        index=len(delivery_days) - 1,
        format_func=lambda day: day.strftime("%A, %d %B %Y"),
        key=f"production_delivery_day_{zone_id}",
    )

actual = period_forecast[actual_column]
p10 = period_forecast[p10_column]
p50 = period_forecast[p50_column]
p90 = period_forecast[p90_column]
absolute_error = (actual - p50).abs()
bias_error = p50 - actual

mae_mw = absolute_error.mean()
rmse_mw = ((actual - p50) ** 2).mean() ** 0.5
nmae_percent = mae_mw / installed_capacity_mw * 100
coverage_percent = actual.between(p10, p90).mean() * 100
mean_interval_width_mw = (p90 - p10).mean()
bias_mw = bias_error.mean()

st.caption(
    f"Accuracy metrics cover {period_start:%d/%m/%Y}–{period_end:%d/%m/%Y}. "
    "The selected day will control the intraday chart."
)

metric_columns = st.columns(6)
metrics = [
    ("MAE", f"{mae_mw:,.2f} MW"),
    ("RMSE", f"{rmse_mw:,.2f} MW"),
    ("nMAE", f"{nmae_percent:.2f}%"),
    ("P10–P90 Coverage", f"{coverage_percent:.1f}%"),
    ("Mean Band Width", f"{mean_interval_width_mw:,.2f} MW"),
    ("P50 Bias", f"{bias_mw:+,.2f} MW"),
]

for column, (label, value) in zip(metric_columns, metrics):
    column.metric(
        label,
        value,
        help=(
            "P50 forecast minus actual. Positive means overforecast."
            if label == "P50 Bias"
            else None
        ),
    )

day_forecast = period_forecast.loc[
    period_forecast["delivery_start_local"].dt.date == selected_day
].copy()

# Keep the interval band separate from the two lines so the tooltip can show all values.
line_data = day_forecast[
    ["delivery_start_local", actual_column, p50_column]
].melt(
    id_vars="delivery_start_local",
    var_name="series",
    value_name="power_mw",
)
line_data["series"] = line_data["series"].map(
    {
        actual_column: "Actual production",
        p50_column: "P50 forecast",
    }
)

delivery_time_axis = alt.X(
    "delivery_start_local:T",
    title="Delivery time",
    axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
)
power_axis = alt.Y(
    f"{p10_column}:Q",
    title="Power (MW)",
    scale=alt.Scale(zero=True),
)

interval_band = alt.Chart(day_forecast).mark_area(
    color="#6FAF8E",
    opacity=0.2,
).encode(
    x=delivery_time_axis,
    y=power_axis,
    y2=f"{p90_column}:Q",
    tooltip=[
        alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
        alt.Tooltip(actual_column, title="Actual production (MW)", format=".1f"),
        alt.Tooltip(p50_column, title="P50 forecast (MW)", format=".1f"),
        alt.Tooltip(p10_column, title="P10 (MW)", format=".1f"),
        alt.Tooltip(p90_column, title="P90 (MW)", format=".1f"),
    ],
)
forecast_lines = alt.Chart(line_data).mark_line(strokeWidth=2.2).encode(
    x=delivery_time_axis,
    y=alt.Y("power_mw:Q", title="Power (MW)", scale=alt.Scale(zero=True)),
    color=alt.Color(
        "series:N",
        title=None,
        sort=["Actual production", "P50 forecast"],
        scale=alt.Scale(
            domain=["Actual production", "P50 forecast"],
            range=["#E7EBF0", "#6FAF8E"],
        ),
        legend=alt.Legend(orient="top"),
    ),
    tooltip=[
        alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("power_mw:Q", title="Power (MW)", format=".1f"),
    ],
)

production_chart = (
    alt.layer(interval_band, forecast_lines)
    .properties(height=400)
    .configure(background="#151E2A")
    .configure_view(strokeOpacity=0)
    .configure_axis(
        gridColor="#29394B",
        labelColor="#AEBBC9",
        titleColor="#D7DFE8",
        domainColor="#34465C",
        tickColor="#34465C",
    )
    .configure_legend(labelColor="#D7DFE8", titleColor="#D7DFE8")
)

with chart_card(
    title=f"{selected_asset_name} production forecast · {selected_day:%d %b %Y}",
    description=(
        "P10–P90 range, P50 forecast and actual virtual production at 15-minute resolution."
    ),
    card_id="production-forecast",
):
    st.altair_chart(production_chart, use_container_width=True)

# Compare P50 with actual production across the full selected period.
diagnostic_data = period_forecast[
    ["delivery_start_local", actual_column, p50_column]
].copy()
diagnostic_data["absolute_error_mw"] = (
    diagnostic_data[actual_column] - diagnostic_data[p50_column]
).abs()
diagnostic_data["bias_mw"] = diagnostic_data[p50_column] - diagnostic_data[actual_column]
diagnostic_data["delivery_day"] = diagnostic_data["delivery_start_local"].dt.date
diagnostic_data["delivery_hour"] = diagnostic_data["delivery_start_local"].dt.hour
diagnostic_data["hour_label"] = diagnostic_data["delivery_hour"].map(
    lambda hour: f"{hour:02d}:00"
)

hour_sort = [f"{hour:02d}:00" for hour in range(24)]
hour_ticks = [f"{hour:02d}:00" for hour in range(0, 24, 2)]
hourly_mae = diagnostic_data.groupby(
    "hour_label", as_index=False
).agg(
    mae_mw=("absolute_error_mw", "mean"),
    bias_mw=("bias_mw", "mean"),
)
hourly_mae["bias_zero"] = 0.0
daily_hourly_error = diagnostic_data.groupby(
    ["delivery_day", "hour_label"], as_index=False
).agg(
    mae_mw=("absolute_error_mw", "mean"),
    bias_mw=("bias_mw", "mean"),
)
day_labels = [day.strftime("%d %b %Y") for day in sorted(delivery_days)]
daily_hourly_error["delivery_day_label"] = daily_hourly_error["delivery_day"].map(
    lambda day: day.strftime("%d %b %Y")
)
day_ticks = day_labels[:: max(1, len(day_labels) // 9)]
if day_ticks[-1] != day_labels[-1]:
    day_ticks.append(day_labels[-1])

hour_axis = alt.X(
    "hour_label:O",
    title="Local delivery hour",
    sort=hour_sort,
    axis=alt.Axis(values=hour_ticks, labelAngle=0),
)
mae_bars = alt.Chart(hourly_mae).mark_bar(
    color="#6FAF8E",
    opacity=0.55,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3,
).encode(
    x=hour_axis,
    y=alt.Y("mae_mw:Q", title="Error (MW)", scale=alt.Scale(zero=True)),
    tooltip=[
        alt.Tooltip("hour_label:O", title="Local delivery hour"),
        alt.Tooltip("mae_mw:Q", title="MAE (MW)", format=".2f"),
    ],
)
bias_line = alt.Chart(hourly_mae).mark_line(
    color="#D9B36C",
    point=True,
    strokeWidth=2,
).encode(
    x=hour_axis,
    y=alt.Y("bias_mw:Q", title="Error (MW)"),
    tooltip=[
        alt.Tooltip("hour_label:O", title="Local delivery hour"),
        alt.Tooltip("bias_mw:Q", title="Bias (MW)", format="+.2f"),
    ],
)
zero_line = alt.Chart(hourly_mae).mark_rule(
    color="#9EADBF",
    strokeDash=[4, 4],
).encode(y="bias_zero:Q")
hourly_error_chart = (
    alt.layer(mae_bars, bias_line, zero_line)
    .resolve_scale(y="shared")
    .properties(height=280)
    .configure(background="#151E2A")
    .configure_view(strokeOpacity=0)
    .configure_axis(
        gridColor="#29394B",
        labelColor="#AEBBC9",
        titleColor="#D7DFE8",
        domainColor="#34465C",
        tickColor="#34465C",
    )
)

daily_error_heatmap = (
    alt.Chart(daily_hourly_error)
    .mark_rect()
    .encode(
        x=hour_axis,
        y=alt.Y(
            "delivery_day_label:O",
            title="Delivery day",
            sort=day_labels,
            axis=alt.Axis(values=day_ticks, labelAngle=0),
        ),
        color=alt.Color(
            "mae_mw:Q",
            title="MAE (MW)",
            scale=alt.Scale(scheme="greens"),
        ),
        tooltip=[
            alt.Tooltip("delivery_day_label:O", title="Delivery day"),
            alt.Tooltip("hour_label:O", title="Local delivery hour"),
            alt.Tooltip("mae_mw:Q", title="MAE (MW)", format=".2f"),
            alt.Tooltip("bias_mw:Q", title="Bias (MW)", format="+.2f"),
        ],
    )
    .properties(height=280)
    .configure(background="#151E2A")
    .configure_view(strokeOpacity=0)
    .configure_axis(
        gridColor="#29394B",
        labelColor="#AEBBC9",
        titleColor="#D7DFE8",
        domainColor="#34465C",
        tickColor="#34465C",
    )
    .configure_legend(labelColor="#D7DFE8", titleColor="#D7DFE8")
)

hourly_column, heatmap_column = st.columns(2, gap="medium")
with hourly_column:
    with chart_card(
        title="P50 error and bias by delivery hour",
        description=(
            "Bars show MAE; the line shows bias (P50 minus actual). "
            "Positive bias means overforecast."
        ),
        card_id="production-error-by-hour",
    ):
        st.altair_chart(hourly_error_chart, use_container_width=True)

with heatmap_column:
    with chart_card(
        title="Daily P50 error heatmap",
        description=(
            "Color shows MAE; hover also shows bias (P50 minus actual). "
            "Positive bias means overforecast."
        ),
        card_id="production-daily-error-heatmap",
    ):
        st.altair_chart(daily_error_heatmap, use_container_width=True)
