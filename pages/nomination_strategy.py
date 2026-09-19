import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.components import chart_card
from src.dashboard.data import filter_period, load_table, selected_market_zone_id


CANDIDATE_LABELS = {
    "p10_prudente": "P10",
    "p50_centrale": "P50",
    "p90_agressive": "P90",
}
CANDIDATE_ORDER = ["P10", "P50", "P90"]
CANDIDATE_COLORS = ["#4F8FCB", "#6FAF8E", "#9B8AC4"]
SELECTED_COLOR = "#D9B36C"
HOUR_ORDER = [f"{hour:02d}:00" for hour in range(24)]
HOUR_TICKS = [f"{hour:02d}:00" for hour in range(0, 24, 2)]


def style_chart(chart):
    """Apply the dashboard's dark chart colors."""
    return (
        chart.configure(background="#151E2A")
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


def candidate_color():
    return alt.Color(
        "candidate:N",
        title=None,
        sort=CANDIDATE_ORDER,
        scale=alt.Scale(domain=CANDIDATE_ORDER, range=CANDIDATE_COLORS),
        legend=alt.Legend(orient="top"),
    )


market_zone = selected_market_zone_id()
period_start, period_end = st.session_state["delivery_period"]

st.title("Nomination Strategy")

scores = filter_period(load_table("scores"), period_start, period_end)
decisions = filter_period(load_table("decisions"), period_start, period_end)

if scores.empty or decisions.empty:
    st.info("No nomination scores are available for the selected period.")
    st.stop()

scores["candidate"] = scores["strategy_name"].map(CANDIDATE_LABELS)
decisions["candidate"] = decisions["strategy_name"].map(CANDIDATE_LABELS)
delivery_days = sorted(scores["delivery_start_local"].dt.date.unique())

st.caption(
    f"{market_zone} · Candidate scores use production and price forecasts. "
    "Expected market spreads are estimated from realised historical data before each "
    "delivery day; that delivery day's settlement does not enter its score."
)

# Show the overall choice frequency before looking at its hourly pattern.
choice_totals = decisions["candidate"].value_counts().reindex(CANDIDATE_ORDER, fill_value=0)
choice_metrics = st.columns(3)
for column, candidate in zip(choice_metrics, CANDIDATE_ORDER):
    count = int(choice_totals[candidate])
    share = count / len(decisions)
    column.metric(
        f"{candidate} choices",
        f"{share:.1%}",
        help=f"Selected in {count:,} of {len(decisions):,} delivery intervals.",
    )

p50_count = int(choice_totals["P50"])
if p50_count == 0:
    p10_share = choice_totals["P10"] / len(decisions)
    p90_share = choice_totals["P90"] / len(decisions)
    if choice_totals["P10"] and choice_totals["P90"]:
        message = (
            "P50 is never selected in the current period. The rule behaves mainly "
            f"as P10 ({p10_share:.1%}), with occasional switches to P90 ({p90_share:.1%})."
        )
    else:
        used_candidate = "P10" if choice_totals["P10"] else "P90"
        message = (
            f"P50 is never selected in the current period. "
            f"All decisions are assigned to {used_candidate}."
        )
    st.info(message)

# Aggregate choices by local delivery hour; each bar sums to 100%.
decisions["delivery_hour"] = decisions["delivery_start_local"].dt.hour.map(
    lambda hour: f"{hour:02d}:00"
)
choice_counts = (
    decisions.groupby(["delivery_hour", "candidate"], as_index=False)
    .size()
    .rename(columns={"size": "interval_count"})
)
hour_totals = choice_counts.groupby("delivery_hour")["interval_count"].transform("sum")
choice_counts["share"] = choice_counts["interval_count"] / hour_totals

# Compare average candidate scores and the gap between the top two scores.
scores["delivery_hour"] = scores["delivery_start_local"].dt.hour.map(
    lambda hour: f"{hour:02d}:00"
)
hourly_scores = (
    scores.groupby(["delivery_hour", "candidate"], as_index=False)
    .agg(mean_score_eur=("expected_pnl_eur", "mean"))
)
score_by_interval = scores.pivot(
    index="delivery_start_utc", columns="candidate", values="expected_pnl_eur"
)
sorted_scores = np.sort(score_by_interval[CANDIDATE_ORDER].to_numpy(), axis=1)
score_by_interval["score_margin_eur"] = sorted_scores[:, -1] - sorted_scores[:, -2]
interval_hours = scores[
    ["delivery_start_utc", "delivery_hour"]
].drop_duplicates("delivery_start_utc")
hourly_margin = (
    score_by_interval[["score_margin_eur"]]
    .reset_index()
    .merge(interval_hours, on="delivery_start_utc", validate="one_to_one")
    .groupby("delivery_hour", as_index=False)
    .agg(mean_margin_eur=("score_margin_eur", "mean"))
)

st.subheader("Nomination choices by delivery hour")
choice_column, score_column = st.columns(2, vertical_alignment="top")

with choice_column:
    with chart_card(
        title="P10/P50/P90 choices by hour",
        description="Relative frequency; each bar totals 100%.",
        card_id="nomination-hourly-choice-frequency",
    ):
        choice_chart = alt.Chart(choice_counts).mark_bar().encode(
            x=alt.X(
                "delivery_hour:O",
                title="Local delivery hour",
                sort=HOUR_ORDER,
                axis=alt.Axis(values=HOUR_TICKS, labelAngle=0),
            ),
            y=alt.Y(
                "share:Q",
                title="Share of decisions",
                stack="zero",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=candidate_color(),
            order=alt.Order("candidate:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("delivery_hour:O", title="Delivery hour"),
                alt.Tooltip("candidate:N", title="Selected candidate"),
                alt.Tooltip("interval_count:Q", title="Intervals"),
                alt.Tooltip("share:Q", title="Frequency", format=".1%"),
            ],
        ).properties(height=330)
        st.altair_chart(style_chart(choice_chart), use_container_width=True)

with score_column:
    with chart_card(
        title="Scores and decision margin by hour",
        description="Mean score above; gap between the best and runner-up below.",
        card_id="nomination-hourly-scores",
    ):
        score_lines = alt.Chart(hourly_scores).mark_line(
            point=True,
            strokeWidth=2,
        ).encode(
            x=alt.X(
                "delivery_hour:O",
                title="Local delivery hour",
                sort=HOUR_ORDER,
                axis=alt.Axis(values=HOUR_TICKS, labelAngle=0),
            ),
            y=alt.Y("mean_score_eur:Q", title="Expected PnL (€/15 min)"),
            color=candidate_color(),
            tooltip=[
                alt.Tooltip("delivery_hour:O", title="Delivery hour"),
                alt.Tooltip("candidate:N", title="Candidate"),
                alt.Tooltip("mean_score_eur:Q", title="Mean score (€)", format=".2f"),
            ],
        ).properties(height=215)

        margin_bars = alt.Chart(hourly_margin).mark_bar(
            color=SELECTED_COLOR,
            opacity=0.8,
        ).encode(
            x=alt.X(
                "delivery_hour:O",
                title=None,
                sort=HOUR_ORDER,
                axis=alt.Axis(values=HOUR_TICKS, labelAngle=0),
            ),
            y=alt.Y(
                "mean_margin_eur:Q",
                title="Best-vs-second score gap (€)",
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip("delivery_hour:O", title="Delivery hour"),
                alt.Tooltip(
                    "mean_margin_eur:Q",
                    title="Mean score gap (€)",
                    format=".2f",
                ),
            ],
        ).properties(height=107)

        hourly_chart = alt.vconcat(score_lines, margin_bars, spacing=8).resolve_scale(
            x="shared"
        )
        st.altair_chart(style_chart(hourly_chart), use_container_width=True)

st.subheader("Daily nomination detail")
day_picker, _ = st.columns([1.2, 4.8])
with day_picker:
    selected_day = st.selectbox(
        "Delivery day",
        options=delivery_days,
        index=len(delivery_days) - 1,
        format_func=lambda day: day.strftime("%d/%m/%Y"),
        key=f"nomination_delivery_day_{market_zone}",
    )

day_scores = scores.loc[scores["delivery_start_local"].dt.date == selected_day].copy()
day_decisions = decisions.loc[
    decisions["delivery_start_local"].dt.date == selected_day
].copy()
day_scores["selected_candidate"] = day_scores["delivery_start_utc"].map(
    day_decisions.set_index("delivery_start_utc")["candidate"]
)

nomination_column, daily_score_column = st.columns(2)

with nomination_column:
    with chart_card(
        title="Candidate and selected nominations",
        description="Candidate quantities by quarter hour; amber shows the selected nomination.",
        card_id="nomination-daily-quantities",
    ):
        candidate_nominations = alt.Chart(day_scores).mark_line(
            interpolate="step-after",
            strokeWidth=1.8,
        ).encode(
            x=alt.X(
                "delivery_start_local:T",
                title="Delivery time",
                axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
            ),
            y=alt.Y(
                "nomination_mwh:Q",
                title="Nomination (MWh / 15 min)",
                scale=alt.Scale(zero=True),
            ),
            color=candidate_color(),
            tooltip=[
                alt.Tooltip(
                    "delivery_start_local:T",
                    title="Delivery time",
                    format="%H:%M",
                ),
                alt.Tooltip("candidate:N", title="Candidate"),
                alt.Tooltip("nomination_mwh:Q", title="Nomination (MWh)", format=".2f"),
            ],
        )
        selected_nomination = alt.Chart(day_decisions).mark_line(
            interpolate="step-after",
            color=SELECTED_COLOR,
            strokeWidth=3.2,
        ).encode(
            x="delivery_start_local:T",
            y="nomination_mwh:Q",
            tooltip=[
                alt.Tooltip(
                    "delivery_start_local:T",
                    title="Delivery time",
                    format="%H:%M",
                ),
                alt.Tooltip("candidate:N", title="Selected candidate"),
                alt.Tooltip("nomination_mwh:Q", title="Selected nomination (MWh)", format=".2f"),
            ],
        )
        nomination_chart = alt.layer(
            candidate_nominations, selected_nomination
        ).properties(height=330)
        st.altair_chart(style_chart(nomination_chart), use_container_width=True)

with daily_score_column:
    with chart_card(
        title="Candidate expected scores",
        description="Expected PnL by quarter hour for each candidate nomination.",
        card_id="nomination-daily-scores",
    ):
        daily_score_chart = alt.Chart(day_scores).mark_line(
            interpolate="step-after",
            point=True,
            strokeWidth=2,
        ).encode(
            x=alt.X(
                "delivery_start_local:T",
                title="Delivery time",
                axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
            ),
            y=alt.Y("expected_pnl_eur:Q", title="Expected PnL (€ / 15 min)"),
            color=candidate_color(),
            tooltip=[
                alt.Tooltip(
                    "delivery_start_local:T",
                    title="Delivery time",
                    format="%H:%M",
                ),
                alt.Tooltip("candidate:N", title="Candidate"),
                alt.Tooltip(
                    "expected_da_revenue_eur:Q",
                    title="Expected DA revenue (€)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "expected_shortfall_mwh:Q",
                    title="Expected shortfall (MWh)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "expected_surplus_mwh:Q",
                    title="Expected surplus (MWh)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "expected_imbalance_settlement_eur:Q",
                    title="Expected imbalance settlement (€)",
                    format=".2f",
                ),
                alt.Tooltip("expected_pnl_eur:Q", title="Expected PnL (€)", format=".2f"),
                alt.Tooltip("selected_candidate:N", title="Selected candidate"),
            ],
        ).properties(height=330)
        st.altair_chart(style_chart(daily_score_chart), use_container_width=True)

# Show the financial decomposition behind the three candidates for the selected day.
daily_candidate_summary = (
    day_scores.groupby("candidate", as_index=False)
    .agg(
        expected_da_revenue_eur=("expected_da_revenue_eur", "sum"),
        expected_shortfall_mwh=("expected_shortfall_mwh", "sum"),
        expected_surplus_mwh=("expected_surplus_mwh", "sum"),
        expected_imbalance_settlement_eur=(
            "expected_imbalance_settlement_eur",
            "sum",
        ),
        expected_pnl_eur=("expected_pnl_eur", "sum"),
    )
    .set_index("candidate")
    .reindex(CANDIDATE_ORDER)
    .reset_index()
    .rename(
        columns={
            "candidate": "Candidate",
            "expected_da_revenue_eur": "Expected DA revenue (€)",
            "expected_shortfall_mwh": "Expected shortfall (MWh)",
            "expected_surplus_mwh": "Expected surplus (MWh)",
            "expected_imbalance_settlement_eur": "Expected imbalance settlement (€)",
            "expected_pnl_eur": "Expected PnL (€)",
        }
    )
)
st.caption("Expected financial decomposition for the selected delivery day")
st.dataframe(
    daily_candidate_summary,
    hide_index=True,
    width="stretch",
    column_config={
        column: st.column_config.NumberColumn(format="%.2f")
        for column in daily_candidate_summary.columns[1:]
    },
)

# Keep the full quarter-hour detail available without crowding the page.
candidate_table = day_scores.pivot(
    index="delivery_start_utc",
    columns="candidate",
    values=["nomination_mwh", "expected_pnl_eur"],
)
candidate_table.columns = [f"{measure}_{candidate}" for measure, candidate in candidate_table.columns]
candidate_table = candidate_table.reset_index()
daily_table = day_decisions[
    [
        "delivery_start_utc",
        "delivery_start_local",
        "candidate",
        "nomination_mwh",
        "expected_da_revenue_eur",
        "expected_shortfall_mwh",
        "expected_surplus_mwh",
        "expected_imbalance_settlement_eur",
        "expected_pnl_eur",
    ]
].merge(candidate_table, on="delivery_start_utc", validate="one_to_one")
daily_table = daily_table.merge(
    score_by_interval["score_margin_eur"].reset_index(),
    on="delivery_start_utc",
    validate="one_to_one",
)
daily_table = daily_table.rename(
    columns={
        "delivery_start_local": "Delivery time",
        "candidate": "Selected candidate",
        "nomination_mwh": "Selected nomination (MWh)",
        "expected_da_revenue_eur": "Selected DA revenue (€)",
        "expected_shortfall_mwh": "Selected shortfall (MWh)",
        "expected_surplus_mwh": "Selected surplus (MWh)",
        "expected_imbalance_settlement_eur": "Selected imbalance settlement (€)",
        "expected_pnl_eur": "Selected score (€)",
        "score_margin_eur": "Margin to runner-up (€)",
        "nomination_mwh_P10": "P10 nomination (MWh)",
        "nomination_mwh_P50": "P50 nomination (MWh)",
        "nomination_mwh_P90": "P90 nomination (MWh)",
        "expected_pnl_eur_P10": "P10 score (€)",
        "expected_pnl_eur_P50": "P50 score (€)",
        "expected_pnl_eur_P90": "P90 score (€)",
    }
)
table_columns = [
    "Delivery time",
    "P10 nomination (MWh)",
    "P50 nomination (MWh)",
    "P90 nomination (MWh)",
    "Selected candidate",
    "Selected nomination (MWh)",
    "Selected DA revenue (€)",
    "Selected shortfall (MWh)",
    "Selected surplus (MWh)",
    "Selected imbalance settlement (€)",
    "Selected score (€)",
    "Margin to runner-up (€)",
    "P10 score (€)",
    "P50 score (€)",
    "P90 score (€)",
]

with st.expander(f"Show all {len(daily_table)} quarter-hour decisions"):
    st.dataframe(
        daily_table[table_columns],
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(format="%.2f")
            for column in table_columns[1:4] + table_columns[5:]
        },
    )
