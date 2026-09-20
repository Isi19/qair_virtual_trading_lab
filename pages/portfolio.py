"""Portfolio summary for the selected historical delivery period."""

import streamlit as st
import altair as alt
import pandas as pd

from src.dashboard.data import (
    filter_period,
    load_table,
    market_zone_config,
    require_available_zone,
)
from src.dashboard.components import chart_card


require_available_zone()
zone_config = market_zone_config()
zone_assets = zone_config["assets"]
st.title("Portfolio Overview")
asset_summary = "  |  ".join(
    f"{asset['name']} · {asset['technology']} · {asset['capacity_mw']:g} MW"
    for asset in zone_assets
)
total_capacity_mw = sum(asset["capacity_mw"] for asset in zone_assets)
st.caption(f"{asset_summary}  |  Total · {total_capacity_mw:.1f} MW")

period_start, period_end = st.session_state["delivery_period"]
policies = [
    ("P10 · Conservative", "always_p10_prudente"),
    ("P50 · Central", "always_p50_centrale"),
    ("P90 · Aggressive", "always_p90_agressive"),
    ("Dynamic", "dynamic_expected_pnl"),
]
DYNAMIC_INFO_TEXT = (
    "At each 15-minute interval, Dynamic compares the estimated PnL of the "
    "P10, P50 and P90 nominations using the available forecasts and assumptions, "
    "then selects the option with the highest estimate. This decision rule does "
    "not guarantee the highest realized PnL."
)


def set_nomination_policy(policy_name: str) -> None:
    """Keep the selected nomination policy when the page reruns."""
    st.session_state["portfolio_nomination_policy"] = policy_name


if "portfolio_nomination_policy" not in st.session_state:
    st.session_state["portfolio_nomination_policy"] = "dynamic_expected_pnl"

st.markdown("**Nomination policy**")
policy_row, _ = st.columns([2.5, 1.5])
policy_columns = policy_row.columns(4)
for column, (label, policy_name) in zip(policy_columns, policies):
    is_selected = st.session_state["portfolio_nomination_policy"] == policy_name
    if policy_name == "dynamic_expected_pnl":
        card_state = "selected" if is_selected else "idle"
        with column.container(
            key=f"policy-card-dynamic-{card_state}",
            gap="small",
        ):
            button_column, info_column = st.columns(
                [0.84, 0.16],
                gap="small",
                vertical_alignment="center",
            )
            button_column.button(
                label,
                key=f"policy_{policy_name}",
                on_click=set_nomination_policy,
                args=(policy_name,),
                type="tertiary",
                use_container_width=True,
            )
            with info_column:
                with st.popover(
                    "ⓘ",
                    help="About the Dynamic rule",
                ):
                    st.write(DYNAMIC_INFO_TEXT)
    else:
        column.button(
            label,
            key=f"policy_{policy_name}",
            on_click=set_nomination_policy,
            args=(policy_name,),
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        )

selected_policy = st.session_state["portfolio_nomination_policy"]

portfolio = filter_period(load_table("portfolio"), period_start, period_end)
settlement = filter_period(load_table("settlement"), period_start, period_end)
settlement = settlement.loc[settlement["policy_name"] == selected_policy]

# These columns already represent energy for each 15-minute interval in MWh.
actual_production_mwh = portfolio["portfolio_actual_mwh"].sum()
p50_forecast_mwh = portfolio["portfolio_p50_mwh"].sum()
nominated_energy_mwh = settlement["nomination_mwh"].sum()
absolute_imbalance_mwh = settlement["imbalance_volume_mwh"].abs().sum()

st.caption("The selected policy affects nomination and imbalance, not production forecasts. \n"
           f"KPIs cover {period_start:%A, %d %B %Y}–{period_end:%A, %d %B %Y}. ")
kpi_columns = st.columns(4)
kpi_values = [
    ("Actual Production", actual_production_mwh),
    ("P50 Forecast", p50_forecast_mwh),
    ("Nominated Energy", nominated_energy_mwh),
    ("Abs. Imbalance", absolute_imbalance_mwh),
]

for column, (label, value) in zip(kpi_columns, kpi_values):
    column.metric(label, f"{value:,.0f} MWh")


delivery_days = sorted(portfolio["delivery_start_local"].dt.date.unique())
day_picker, _ = st.columns([1.1, 3.9])
with day_picker:
    selected_day = st.selectbox(
        "Delivery day",
        options=delivery_days,
        index=len(delivery_days) - 1,
        format_func=lambda day: day.strftime("%A, %d %B %Y"),
        key="portfolio_chart_delivery_day",
    )

day_portfolio = portfolio.loc[
    portfolio["delivery_start_local"].dt.date == selected_day
].copy()
day_settlement = settlement.loc[
    settlement["delivery_start_local"].dt.date == selected_day,
    ["delivery_start_utc", "nomination_mwh"],
]
day_chart = day_portfolio.merge(
    day_settlement,
    on="delivery_start_utc",
    validate="one_to_one",
)

# Forecasts are in MW; convert nominated quarter-hour energy back to average MW.
day_chart["nomination_mw"] = day_chart["nomination_mwh"] / 0.25
chart_series = day_chart.melt(
    id_vars="delivery_start_local",
    value_vars=["portfolio_actual_mw", "portfolio_p50_mw", "nomination_mw"],
    var_name="series",
    value_name="power_mw",
)
chart_series["series"] = chart_series["series"].map(
    {
        "portfolio_actual_mw": "Actual production",
        "portfolio_p50_mw": "P50 forecast",
        "nomination_mw": "Nomination",
    }
)
legend_series = pd.concat(
    [
        chart_series,
        pd.DataFrame(
            {
                "delivery_start_local": [day_chart["delivery_start_local"].iloc[0]],
                "series": ["P10–P90 range"],
                "power_mw": [float("nan")],
            }
        ),
    ],
    ignore_index=True,
)

x_axis = alt.X(
    "delivery_start_local:T",
    title="Delivery time",
    axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
)
power_axis = alt.Y(
    "portfolio_p10_mw:Q",
    title="Power (MW)",
    scale=alt.Scale(zero=True),
)
legend_domain = [
    "Actual production",
    "P50 forecast",
    "Nomination",
    "P10–P90 range",
]
legend_colors = ["#E7EBF0", "#6FAF8E", "#D9B36C", "#9BC8AE"]
uncertainty_band = alt.Chart(day_chart).mark_area(
    color="#6FAF8E",
    opacity=0.2,
).encode(
    x=x_axis,
    y=power_axis,
    y2="portfolio_p90_mw:Q",
    tooltip=[
        alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
        alt.Tooltip("portfolio_actual_mw:Q", title="Actual production (MW)", format=".1f"),
        alt.Tooltip("portfolio_p50_mw:Q", title="P50 (MW)", format=".1f"),
        alt.Tooltip("nomination_mw:Q", title="Nomination (MW)", format=".1f"),
        alt.Tooltip("portfolio_p10_mw:Q", title="P10 (MW)", format=".1f"),
        alt.Tooltip("portfolio_p90_mw:Q", title="P90 (MW)", format=".1f"),
    ],
)
production_lines = alt.Chart(legend_series).mark_line(strokeWidth=2.2).encode(
    x=x_axis,
    y=alt.Y("power_mw:Q", title="Power (MW)", scale=alt.Scale(zero=True)),
    color=alt.Color(
        "series:N",
        title=None,
        sort=legend_domain,
        scale=alt.Scale(domain=legend_domain, range=legend_colors),
        legend=alt.Legend(orient="top"),
    ),
    tooltip=[
        alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("power_mw:Q", title="Power (MW)", format=".1f"),
    ],
)

portfolio_chart = (
    alt.layer(uncertainty_band, production_lines)
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
    title=f"Production forecast and nomination · {selected_day:%d %B %Y}",
    description=(
        "The shaded area shows the P10–P90 range; nomination is displayed as average MW "
        "for each 15-minute interval."
    ),
    card_id="portfolio-production",
):
    st.altair_chart(portfolio_chart, use_container_width=True)

contribution_columns = [
    asset["portfolio_actual_column"] for asset in zone_assets
]
contribution_data = day_chart.melt(
    id_vars="delivery_start_local",
    value_vars=contribution_columns,
    var_name="actual_column",
    value_name="power_mw",
)
asset_by_column = {
    asset["portfolio_actual_column"]: asset for asset in zone_assets
}
contribution_data["asset"] = contribution_data["actual_column"].map(
    {
        column: asset["name"]
        for column, asset in asset_by_column.items()
    }
)
contribution_data["stack_order"] = contribution_data["asset"].map(
    {asset["name"]: asset["stack_order"] for asset in zone_assets}
)
chart_assets = sorted(zone_assets, key=lambda asset: asset["stack_order"])
asset_names = [asset["name"] for asset in chart_assets]
asset_colors = [asset["color"] for asset in chart_assets]
technology_names = list(dict.fromkeys(asset["technology"] for asset in zone_assets))
contribution_title = " / ".join(technology_names) + " contribution"

contribution_chart = (
    alt.Chart(contribution_data)
    .mark_area(opacity=0.82)
    .encode(
        x=x_axis,
        y=alt.Y(
            "power_mw:Q",
            title="Power (MW)",
            stack="zero",
            scale=alt.Scale(zero=True),
        ),
        color=alt.Color(
            "asset:N",
            title=None,
            sort=asset_names,
            scale=alt.Scale(
                domain=asset_names,
                range=asset_colors,
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        order=alt.Order("stack_order:Q", sort="ascending"),
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("asset:N", title="Asset"),
            alt.Tooltip("power_mw:Q", title="Power (MW)", format=".1f"),
        ],
    )
    .properties(height=260)
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
contribution_column, decision_column = st.columns(2, gap="medium")
with contribution_column:
    with chart_card(
        title=contribution_title,
        description="Stacked 15-minute production for the selected delivery day.",
        card_id="wind-solar-contribution",
    ):
        st.altair_chart(contribution_chart, use_container_width=True)

# This shows which production quantile the dynamic rule selected per interval.
decisions = filter_period(load_table("decisions"), period_start, period_end)
decision_labels = {
    "p10_prudente": "P10",
    "p50_centrale": "P50",
    "p90_agressive": "P90",
}
choice_counts = (
    decisions["strategy_name"]
    .map(decision_labels)
    .value_counts()
    .reindex(["P10", "P50", "P90"], fill_value=0)
)
choice_distribution = pd.DataFrame(
    {
        "quantile": choice_counts.index,
        "interval_count": choice_counts.values,
    }
)
choice_distribution["share_pct"] = (
    choice_distribution["interval_count"] / choice_distribution["interval_count"].sum() * 100
)
choice_distribution["share_label"] = choice_distribution["share_pct"].map(
    lambda share: f"{share:.1f}%"
)

choice_bars = alt.Chart(choice_distribution).mark_bar(
    size=24,
    cornerRadiusEnd=3,
).encode(
    x=alt.X(
        "share_pct:Q",
        title="Share of delivery intervals (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(values=[0, 25, 50, 75, 100], format=".0f"),
    ),
    y=alt.Y(
        "quantile:N",
        title=None,
        sort=["P10", "P50", "P90"],
    ),
    color=alt.Color(
        "quantile:N",
        title=None,
        scale=alt.Scale(
            domain=["P10", "P50", "P90"],
            range=["#4F8FCB", "#6FAF8E", "#D9B36C"],
        ),
        legend=None,
    ),
    tooltip=[
        alt.Tooltip("quantile:N", title="Selected nomination"),
        alt.Tooltip("interval_count:Q", title="15-minute intervals", format=","),
        alt.Tooltip("share_pct:Q", title="Share", format=".1f"),
    ],
)
choice_labels = alt.Chart(choice_distribution).mark_text(
    align="left",
    baseline="middle",
    dx=6,
    color="#E7EBF0",
).encode(
    x="share_pct:Q",
    y=alt.Y("quantile:N", sort=["P10", "P50", "P90"]),
    text="share_label:N",
)
choice_chart = (
    alt.layer(choice_bars, choice_labels)
    .properties(height=260)
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

with decision_column:
    with chart_card(
        title="Dynamic nomination mix",
        description=(
            "Share of 15-minute intervals assigned to each production quantile over "
            "the selected period."
        ),
        card_id="dynamic-nomination-mix",
        info_text=DYNAMIC_INFO_TEXT,
    ):
        st.altair_chart(choice_chart, use_container_width=True)
