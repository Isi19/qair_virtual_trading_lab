import altair as alt
import streamlit as st

from src.dashboard.components import chart_card
from src.dashboard.data import filter_period, load_table, selected_market_zone_id

market_zone = selected_market_zone_id()
period_start, period_end = st.session_state["delivery_period"]

st.title("Day-Ahead Price")

price_data = filter_period(load_table("price"), period_start, period_end)
delivery_days = sorted(price_data["delivery_start_local"].dt.date.unique())

if not delivery_days:
    st.info("No Day-Ahead price forecasts are available for the selected period.")
    st.stop()

actual = price_data["actual_day_ahead_price_eur_mwh"]
p10 = price_data["price_p10_eur_mwh"]
p50 = price_data["price_p50_eur_mwh"]
p90 = price_data["price_p90_eur_mwh"]
absolute_error = (actual - p50).abs()
bias = p50 - actual

mae = absolute_error.mean()
rmse = ((actual - p50) ** 2).mean() ** 0.5
bias_mean = bias.mean()
coverage = actual.between(p10, p90).mean() * 100
mean_interval_width = (p90 - p10).mean()

st.caption(
    f"{market_zone} · Accuracy metrics cover "
    f"{period_start:%d/%m/%Y}–{period_end:%d/%m/%Y}. "
    "Bias is P50 minus actual; positive means overforecast. "
    "Negative prices are retained."
)

metric_columns = st.columns(5)
metrics = [
    ("MAE", f"{mae:,.2f} €/MWh"),
    ("RMSE", f"{rmse:,.2f} €/MWh"),
    ("P50 Bias", f"{bias_mean:+,.2f} €/MWh"),
    ("P10–P90 Coverage", f"{coverage:.1f}%"),
    ("Mean Band Width", f"{mean_interval_width:,.2f} €/MWh"),
]

for column, (label, value) in zip(metric_columns, metrics):
    column.metric(
        label,
        value,
        help=(
            "Observed coverage over the selected period; the nominal P10–P90 target is 80%."
            if label == "P10–P90 Coverage"
            else None
        ),
    )

with chart_card(
    title=f"{market_zone} Day-Ahead price forecast",
    description=(
        "P10–P90 range, P50 forecast and realised price. "
        "The dashed line marks zero; negative prices remain visible."
    ),
    card_id="day-ahead-price-forecast",
):
    # This date selector belongs to the intraday chart, not the period metrics.
    day_column, _ = st.columns([1.2, 3.8])
    with day_column:
        selected_day = st.selectbox(
            "Delivery day",
            options=delivery_days,
            index=len(delivery_days) - 1,
            format_func=lambda day: day.strftime("%A, %d %B %Y"),
            key=f"price_delivery_day_{market_zone}",
        )

    day_price = price_data.loc[
        price_data["delivery_start_local"].dt.date == selected_day
    ].copy()
    line_data = day_price[
        [
            "delivery_start_local",
            "actual_day_ahead_price_eur_mwh",
            "price_p50_eur_mwh",
        ]
    ].melt(
        id_vars="delivery_start_local",
        var_name="series",
        value_name="price_eur_mwh",
    )
    line_data["series"] = line_data["series"].map(
        {
            "actual_day_ahead_price_eur_mwh": "Actual price",
            "price_p50_eur_mwh": "P50 forecast",
        }
    )

    delivery_time_axis = alt.X(
        "delivery_start_local:T",
        title="Delivery time",
        axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
    )
    price_axis = alt.Y(
        "price_p10_eur_mwh:Q",
        title="Day-Ahead price (€/MWh)",
        scale=alt.Scale(zero=False),
    )

    uncertainty_band = alt.Chart(day_price).mark_area(
        color="#6FAF8E",
        opacity=0.2,
    ).encode(
        x=delivery_time_axis,
        y=price_axis,
        y2="price_p90_eur_mwh:Q",
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip(
                "actual_day_ahead_price_eur_mwh:Q",
                title="Actual price (€/MWh)",
                format=".2f",
            ),
            alt.Tooltip("price_p50_eur_mwh:Q", title="P50 (€/MWh)", format=".2f"),
            alt.Tooltip("price_p10_eur_mwh:Q", title="P10 (€/MWh)", format=".2f"),
            alt.Tooltip("price_p90_eur_mwh:Q", title="P90 (€/MWh)", format=".2f"),
        ],
    )
    price_lines = alt.Chart(line_data).mark_line(strokeWidth=2.2).encode(
        x=delivery_time_axis,
        y=alt.Y("price_eur_mwh:Q", title="Day-Ahead price (€/MWh)"),
        color=alt.Color(
            "series:N",
            title=None,
            sort=["Actual price", "P50 forecast"],
            scale=alt.Scale(
                domain=["Actual price", "P50 forecast"],
                range=["#E7EBF0", "#6FAF8E"],
            ),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("price_eur_mwh:Q", title="Price (€/MWh)", format=".2f"),
        ],
    )
    day_price["zero_price"] = 0.0
    zero_line = alt.Chart(day_price).mark_rule(
        color="#D9B36C",
        strokeDash=[5, 4],
        strokeWidth=1.5,
    ).encode(y="zero_price:Q")

    price_chart = (
        alt.layer(uncertainty_band, price_lines, zero_line)
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
    st.altair_chart(price_chart, use_container_width=True)

# Summarise P50 errors over the full selected period, by local delivery hour.
diagnostic_data = price_data[
    ["delivery_start_local", "actual_day_ahead_price_eur_mwh", "price_p50_eur_mwh"]
].copy()
diagnostic_data["absolute_error"] = (
    diagnostic_data["actual_day_ahead_price_eur_mwh"]
    - diagnostic_data["price_p50_eur_mwh"]
).abs()
diagnostic_data["bias"] = (
    diagnostic_data["price_p50_eur_mwh"]
    - diagnostic_data["actual_day_ahead_price_eur_mwh"]
)
diagnostic_data["delivery_day"] = diagnostic_data["delivery_start_local"].dt.date
diagnostic_data["hour_label"] = diagnostic_data["delivery_start_local"].dt.hour.map(
    lambda hour: f"{hour:02d}:00"
)

hour_sort = [f"{hour:02d}:00" for hour in range(24)]
hour_ticks = [f"{hour:02d}:00" for hour in range(0, 24, 2)]
hourly_errors = diagnostic_data.groupby(
    "hour_label", as_index=False
).agg(
    mae=("absolute_error", "mean"),
    bias=("bias", "mean"),
)
hourly_errors["zero"] = 0.0
daily_hourly_errors = diagnostic_data.groupby(
    ["delivery_day", "hour_label"], as_index=False
).agg(
    mae=("absolute_error", "mean"),
    bias=("bias", "mean"),
)
day_labels = [day.strftime("%d %b %Y") for day in delivery_days]
daily_hourly_errors["delivery_day_label"] = daily_hourly_errors[
    "delivery_day"
].map(lambda day: day.strftime("%d %b %Y"))
day_ticks = day_labels[:: max(1, len(day_labels) // 9)]
if day_ticks[-1] != day_labels[-1]:
    day_ticks.append(day_labels[-1])

hour_axis = alt.X(
    "hour_label:O",
    title="Local delivery hour",
    sort=hour_sort,
    axis=alt.Axis(values=hour_ticks, labelAngle=0),
)
mae_bars = alt.Chart(hourly_errors).mark_bar(
    color="#6FAF8E",
    opacity=0.55,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3,
).encode(
    x=hour_axis,
    y=alt.Y("mae:Q", title="Error (€/MWh)", scale=alt.Scale(zero=True)),
    tooltip=[
        alt.Tooltip("hour_label:O", title="Local delivery hour"),
        alt.Tooltip("mae:Q", title="MAE (€/MWh)", format=".2f"),
    ],
)
bias_line = alt.Chart(hourly_errors).mark_line(
    color="#D9B36C",
    point=True,
    strokeWidth=2,
).encode(
    x=hour_axis,
    y=alt.Y("bias:Q", title="Error (€/MWh)"),
    tooltip=[
        alt.Tooltip("hour_label:O", title="Local delivery hour"),
        alt.Tooltip("bias:Q", title="Bias (€/MWh)", format="+.2f"),
    ],
)
zero_line = alt.Chart(hourly_errors).mark_rule(
    color="#9EADBF",
    strokeDash=[4, 4],
).encode(y="zero:Q")
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
    alt.Chart(daily_hourly_errors)
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
            "mae:Q",
            title="MAE (€/MWh)",
            scale=alt.Scale(scheme="greens"),
        ),
        tooltip=[
            alt.Tooltip("delivery_day_label:O", title="Delivery day"),
            alt.Tooltip("hour_label:O", title="Local delivery hour"),
            alt.Tooltip("mae:Q", title="MAE (€/MWh)", format=".2f"),
            alt.Tooltip("bias:Q", title="Bias (€/MWh)", format="+.2f"),
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
            "Positive bias means price overforecast."
        ),
        card_id="day-ahead-price-error-by-hour",
    ):
        st.altair_chart(hourly_error_chart, use_container_width=True)

with heatmap_column:
    with chart_card(
        title="Daily P50 error heatmap",
        description=(
            "Color shows MAE; hover also shows bias (P50 minus actual). "
            "Positive bias means price overforecast."
        ),
        card_id="day-ahead-price-daily-error-heatmap",
    ):
        st.altair_chart(daily_error_heatmap, use_container_width=True)
