"""Compare realised results for the four nomination policies."""

import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

from src.dashboard.components import chart_card
from src.dashboard.data import (
    filter_period,
    load_table,
    require_available_zone,
    selected_market_zone_id,
)


POLICY_LABELS = {
    "always_p10_prudente": "P10 · Conservative",
    "always_p50_centrale": "P50 · Central",
    "always_p90_agressive": "P90 · Aggressive",
    "dynamic_expected_pnl": "Dynamic",
}
POLICY_ORDER = [
    "P10 · Conservative",
    "P50 · Central",
    "P90 · Aggressive",
    "Dynamic",
]
POLICY_COLORS = ["#4F8FCB", "#6FAF8E", "#9B8AC4", "#D9B36C"]
POLICY_SETTLEMENT_NAMES = {
    "P10 · Conservative": "always_p10_prudente",
    "P50 · Central": "always_p50_centrale",
    "P90 · Aggressive": "always_p90_agressive",
    "Dynamic": "dynamic_expected_pnl",
}
CANDIDATE_LABELS = {
    "p10_prudente": "P10",
    "p50_centrale": "P50",
    "p90_agressive": "P90",
}
CANDIDATE_ORDER = ["P10", "P50", "P90"]
CANDIDATE_COLORS = ["#4F8FCB", "#6FAF8E", "#9B8AC4"]


require_available_zone()
market_zone = selected_market_zone_id()
period_start, period_end = st.session_state["delivery_period"]
settlement = filter_period(load_table("settlement"), period_start, period_end)
scores = filter_period(load_table("scores"), period_start, period_end)
decisions = filter_period(load_table("decisions"), period_start, period_end)

st.title("Backtest & Settlement")
st.caption(
    f"{market_zone} · Realised comparison of the four nomination policies "
    f"from {period_start:%A, %d %B %Y} to {period_end:%A, %d %B %Y}."
)


def build_policy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realised financial and volume outcomes by policy."""
    values = frame.copy()

    # A negative imbalance means that the nomination was above production.
    values["shortfall_mwh"] = (-values["imbalance_volume_mwh"]).clip(lower=0)
    values["surplus_mwh"] = values["imbalance_volume_mwh"].clip(lower=0)
    values["absolute_imbalance_mwh"] = values["imbalance_volume_mwh"].abs()

    summary = (
        values.groupby("policy_name", as_index=False)
        .agg(
            gross_pnl_eur=("gross_pnl_eur", "sum"),
            da_revenue_eur=("day_ahead_revenue_eur", "sum"),
            imbalance_settlement_eur=("imbalance_settlement_eur", "sum"),
            absolute_imbalance_mwh=("absolute_imbalance_mwh", "sum"),
            shortfall_mwh=("shortfall_mwh", "sum"),
            surplus_mwh=("surplus_mwh", "sum"),
            signed_imbalance_mwh=("imbalance_volume_mwh", "sum"),
        )
        .assign(policy=lambda result: result["policy_name"].map(POLICY_LABELS))
        .set_index("policy")
        .reindex(POLICY_ORDER)
        .reset_index()
    )
    return summary

if settlement.empty:
    st.info("No realised settlement data is available for the selected period.")
    st.stop()
if scores.empty or decisions.empty:
    st.info("No nomination data is available for the selected period.")
    st.stop()

summary = build_policy_summary(settlement)

with chart_card(
    title="Policy comparison",
    description=(
        "Realised Gross PnL by policy. Hover a bar for the Day-Ahead, imbalance "
        "and volume breakdown."
    ),
    card_id="backtest-policy-comparison",
):
    policy_chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "policy:N",
                title=None,
                sort=POLICY_ORDER,
                axis=alt.Axis(labelAngle=0, labelLimit=140),
            ),
            y=alt.Y("gross_pnl_eur:Q", title="Realised Gross PnL (€)"),
            color=alt.Color(
                "policy:N",
                title="Policy",
                sort=POLICY_ORDER,
                scale=alt.Scale(domain=POLICY_ORDER, range=POLICY_COLORS),
            ),
            tooltip=[
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip("gross_pnl_eur:Q", title="Gross PnL (€)", format=",.2f"),
                alt.Tooltip("da_revenue_eur:Q", title="DA Revenue (€)", format=",.2f"),
                alt.Tooltip(
                    "imbalance_settlement_eur:Q",
                    title="Imbalance Settlement (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "absolute_imbalance_mwh:Q",
                    title="Absolute Imbalance (MWh)",
                    format=",.2f",
                ),
                alt.Tooltip("shortfall_mwh:Q", title="Shortfall (MWh)", format=",.2f"),
                alt.Tooltip("surplus_mwh:Q", title="Surplus (MWh)", format=",.2f"),
                alt.Tooltip(
                    "signed_imbalance_mwh:Q",
                    title="Signed Imbalance (MWh)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=300)
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
    st.altair_chart(policy_chart, use_container_width=True)

    breakdown = summary.rename(
        columns={
            "policy": "Policy",
            "gross_pnl_eur": "Gross PnL (€)",
            "da_revenue_eur": "DA Revenue (€)",
            "imbalance_settlement_eur": "Imbalance Settlement (€)",
            "absolute_imbalance_mwh": "Absolute Imbalance (MWh)",
            "shortfall_mwh": "Shortfall (MWh)",
            "surplus_mwh": "Surplus (MWh)",
            "signed_imbalance_mwh": "Signed Imbalance (MWh)",
        }
    )
    with st.expander("View policy breakdown"):
        st.dataframe(
            breakdown[
                [
                    "Policy",
                    "Gross PnL (€)",
                    "DA Revenue (€)",
                    "Imbalance Settlement (€)",
                    "Absolute Imbalance (MWh)",
                    "Shortfall (MWh)",
                    "Surplus (MWh)",
                    "Signed Imbalance (MWh)",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%,.2f")
                for column in breakdown.columns[1:]
            },
        )

st.caption(
    "Gross PnL = DA Revenue + Imbalance Settlement. "
    "Shortfall and surplus are directional volumes; they cannot be positive "
    "simultaneously for the same interval."
)

# Aggregate realised PnL by local calendar day before calculating the running total.
daily_pnl = (
    settlement.assign(delivery_date=settlement["delivery_start_local"].dt.date)
    .groupby(["delivery_date", "policy_name"], as_index=False)
    .agg(
        daily_gross_pnl_eur=("gross_pnl_eur", "sum"),
        daily_absolute_imbalance_mwh=(
            "imbalance_volume_mwh",
            lambda values: values.abs().sum(),
        ),
    )
    .sort_values(["policy_name", "delivery_date"])
)
daily_pnl["cumulative_gross_pnl_eur"] = daily_pnl.groupby("policy_name")[
    "daily_gross_pnl_eur"
].cumsum()
daily_pnl["policy"] = daily_pnl["policy_name"].map(POLICY_LABELS)

with chart_card(
    title="Cumulative Gross PnL",
    description="Running realised Gross PnL by policy, aggregated by local delivery day.",
    card_id="backtest-cumulative-pnl",
):
    cumulative_chart = (
        alt.Chart(daily_pnl)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("delivery_date:T", title="Delivery date"),
            y=alt.Y("cumulative_gross_pnl_eur:Q", title="Cumulative Gross PnL (€)"),
            color=alt.Color(
                "policy:N",
                title="Policy",
                sort=POLICY_ORDER,
                scale=alt.Scale(domain=POLICY_ORDER, range=POLICY_COLORS),
            ),
            tooltip=[
                alt.Tooltip("delivery_date:T", title="Delivery date", format="%A, %d %B %Y"),
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip(
                    "daily_gross_pnl_eur:Q",
                    title="Daily Gross PnL (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "cumulative_gross_pnl_eur:Q",
                    title="Cumulative Gross PnL (€)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=320)
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
    st.altair_chart(cumulative_chart, use_container_width=True)

with chart_card(
    title="Gross PnL vs Absolute Imbalance",
    description=(
        "Each point is one delivery day and one policy. "
        "The chart shows the realised daily performance-risk trade-off."
    ),
    card_id="backtest-pnl-vs-imbalance",
    info_text=(
        "How to read this chart: each point represents one delivery day and one "
        "policy. The preferred area is the upper-left corner, combining higher "
        "Gross PnL with lower Absolute Imbalance. Points in the lower-right show "
        "a less favourable performance-exposure trade-off."
    ),
):
    pnl_risk_chart = (
        alt.Chart(daily_pnl)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X(
                "daily_absolute_imbalance_mwh:Q",
                title="Absolute Imbalance (MWh)",
            ),
            y=alt.Y("daily_gross_pnl_eur:Q", title="Daily Gross PnL (€)"),
            color=alt.Color(
                "policy:N",
                title="Policy",
                sort=POLICY_ORDER,
                scale=alt.Scale(domain=POLICY_ORDER, range=POLICY_COLORS),
            ),
            tooltip=[
                alt.Tooltip("delivery_date:T", title="Delivery date", format="%A, %d %B %Y"),
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip(
                    "daily_gross_pnl_eur:Q",
                    title="Daily Gross PnL (€)",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "daily_absolute_imbalance_mwh:Q",
                    title="Absolute Imbalance (MWh)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=320)
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
    st.altair_chart(pnl_risk_chart, use_container_width=True)

with chart_card(
    title="Daily Gross PnL",
    description="Realised Gross PnL by local delivery day and policy.",
    card_id="backtest-daily-pnl",
):
    daily_pnl_chart = (
        alt.Chart(daily_pnl)
        .mark_line(point=True, strokeWidth=1.8)
        .encode(
            x=alt.X("delivery_date:T", title="Delivery date"),
            y=alt.Y("daily_gross_pnl_eur:Q", title="Daily Gross PnL (€)"),
            color=alt.Color(
                "policy:N",
                title="Policy",
                sort=POLICY_ORDER,
                scale=alt.Scale(domain=POLICY_ORDER, range=POLICY_COLORS),
            ),
            tooltip=[
                alt.Tooltip("delivery_date:T", title="Delivery date", format="%A, %d %B %Y"),
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip(
                    "daily_gross_pnl_eur:Q",
                    title="Daily Gross PnL (€)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=320)
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
    st.altair_chart(daily_pnl_chart, use_container_width=True)


with chart_card(
    title="Best and worst delivery days",
    description=(
        "Focus on the strongest or weakest realised days for one policy. "
        "The full daily history remains available below."
    ),
    card_id="backtest-extreme-days",
):
    policy_picker, outcome_picker, _ = st.columns([1.15, 1.0, 3.85])
    with policy_picker:
        selected_extreme_policy = st.selectbox(
            "Policy",
            options=POLICY_ORDER,
            index=POLICY_ORDER.index("Dynamic"),
            format_func=lambda policy: policy.split(" · ")[0],
            key=f"backtest_extreme_policy_{market_zone}",
        )
    with outcome_picker:
        selected_outcome = st.radio(
            "Delivery days",
            options=["Top 3", "Bottom 3"],
            horizontal=True,
            key=f"backtest_extreme_outcome_{market_zone}",
        )

    selected_policy_days = daily_pnl.loc[
        daily_pnl["policy"] == selected_extreme_policy
    ].copy()
    if selected_outcome == "Top 3":
        visible_extreme_days = selected_policy_days.nlargest(
            3, "daily_gross_pnl_eur"
        ).assign(outcome="Best")
    else:
        visible_extreme_days = selected_policy_days.nsmallest(
            3, "daily_gross_pnl_eur"
        ).assign(outcome="Worst")

    visible_extreme_days = visible_extreme_days.rename(
        columns={
            "policy": "Policy",
            "outcome": "Outcome",
            "delivery_date": "Delivery date",
            "daily_gross_pnl_eur": "Daily Gross PnL (€)",
            "daily_absolute_imbalance_mwh": "Absolute Imbalance (MWh)",
        }
    )
    st.dataframe(
        visible_extreme_days[
            [
                "Policy",
                "Outcome",
                "Delivery date",
                "Daily Gross PnL (€)",
                "Absolute Imbalance (MWh)",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Delivery date": st.column_config.DateColumn(format="dddd, D MMMM YYYY"),
            "Daily Gross PnL (€)": st.column_config.NumberColumn(format="%.2f"),
            "Absolute Imbalance (MWh)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    full_policy_days = selected_policy_days.rename(
        columns={
            "policy": "Policy",
            "delivery_date": "Delivery date",
            "daily_gross_pnl_eur": "Daily Gross PnL (€)",
            "daily_absolute_imbalance_mwh": "Absolute Imbalance (MWh)",
        }
    )
    with st.expander("View full daily detail"):
        st.dataframe(
            full_policy_days[
                [
                    "Policy",
                    "Delivery date",
                    "Daily Gross PnL (€)",
                    "Absolute Imbalance (MWh)",
                ]
            ].sort_values("Delivery date"),
            hide_index=True,
            width="stretch",
            column_config={
                "Delivery date": st.column_config.DateColumn(format="dddd, D MMMM YYYY"),
                "Daily Gross PnL (€)": st.column_config.NumberColumn(format="%.2f"),
                "Absolute Imbalance (MWh)": st.column_config.NumberColumn(format="%.2f"),
            },
        )


st.subheader("Day Replay")
replay_days = sorted(settlement["delivery_start_local"].dt.date.unique())
replay_picker, _ = st.columns([1.2, 4.8])
with replay_picker:
    replay_day = st.selectbox(
        "Delivery day",
        options=replay_days,
        index=len(replay_days) - 1,
        format_func=lambda day: day.strftime("%A, %d %B %Y"),
        key=f"backtest_replay_day_{market_zone}",
    )

replay_scores = scores.loc[
    scores["delivery_start_local"].dt.date == replay_day
].copy()
replay_decisions = decisions.loc[
    decisions["delivery_start_local"].dt.date == replay_day
].copy()

if replay_scores.empty or replay_decisions.empty:
    st.info("No nomination replay data is available for the selected delivery day.")
    st.stop()

replay_scores["candidate"] = replay_scores["strategy_name"].map(CANDIDATE_LABELS)
replay_decisions["candidate"] = replay_decisions["strategy_name"].map(
    CANDIDATE_LABELS
)

with chart_card(
    title="EX-ANTE · Nomination decision",
    description=(
        "Information available before delivery: candidate nominations, expected "
        "scores and the selected nomination."
    ),
    card_id="backtest-replay-ex-ante",
):
    candidate_lines = alt.Chart(replay_scores).mark_line(
        interpolate="step-after",
        strokeWidth=1.8,
    ).encode(
        x=alt.X("delivery_start_local:T", title="Delivery time"),
        y=alt.Y("nomination_mwh:Q", title="Nomination (MWh / 15 min)"),
        color=alt.Color(
            "candidate:N",
            title="Candidate",
            sort=CANDIDATE_ORDER,
            scale=alt.Scale(domain=CANDIDATE_ORDER, range=CANDIDATE_COLORS),
        ),
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("candidate:N", title="Candidate"),
            alt.Tooltip("nomination_mwh:Q", title="Nomination (MWh)", format=",.2f"),
            alt.Tooltip("expected_pnl_eur:Q", title="Expected PnL (€)", format=",.2f"),
        ],
    )
    selected_line = alt.Chart(replay_decisions).mark_line(
        interpolate="step-after",
        color="#D9B36C",
        strokeWidth=3,
    ).encode(
        x="delivery_start_local:T",
        y="nomination_mwh:Q",
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("candidate:N", title="Selected candidate"),
            alt.Tooltip("nomination_mwh:Q", title="Selected nomination (MWh)", format=",.2f"),
            alt.Tooltip("expected_pnl_eur:Q", title="Selected Expected PnL (€)", format=",.2f"),
        ],
    )
    ex_ante_chart = (
        alt.layer(candidate_lines, selected_line)
        .properties(height=300)
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
    st.altair_chart(ex_ante_chart, use_container_width=True)

    ex_ante_table = replay_scores.pivot(
        index="delivery_start_local",
        columns="candidate",
        values=["nomination_mwh", "expected_pnl_eur"],
    )
    ex_ante_table.columns = [
        f"{measure}_{candidate}" for measure, candidate in ex_ante_table.columns
    ]
    ex_ante_table = ex_ante_table.reset_index().merge(
        replay_decisions[
            [
                "delivery_start_local",
                "candidate",
                "nomination_mwh",
                "expected_pnl_eur",
            ]
        ],
        on="delivery_start_local",
        how="left",
        validate="one_to_one",
        suffixes=("", "_selected"),
    )
    ex_ante_table = ex_ante_table.rename(
        columns={
            "delivery_start_local": "Delivery time",
            "candidate": "Selected candidate",
            "nomination_mwh": "Selected nomination (MWh)",
            "expected_pnl_eur": "Selected Expected PnL (€)",
            "nomination_mwh_P10": "P10 nomination (MWh)",
            "nomination_mwh_P50": "P50 nomination (MWh)",
            "nomination_mwh_P90": "P90 nomination (MWh)",
            "expected_pnl_eur_P10": "P10 Expected PnL (€)",
            "expected_pnl_eur_P50": "P50 Expected PnL (€)",
            "expected_pnl_eur_P90": "P90 Expected PnL (€)",
        }
    )
    ex_ante_table.insert(
        1,
        "Status",
        "Selected · " + ex_ante_table["Selected candidate"],
    )

    def highlight_selected(row):
        return [
            "background-color: rgba(217, 179, 108, 0.12);"
            if column == "Status"
            else ""
            for column in row.index
        ]

    with st.expander("Show EX-ANTE quarter-hour details"):
        st.dataframe(
            ex_ante_table.style.apply(highlight_selected, axis=1),
            hide_index=True,
            width="stretch",
        )

with chart_card(
    title="EX-POST · Realised outcome",
    description=(
        "Information known after delivery: production, realised prices, imbalance "
        "and settlement for the selected replay policy."
    ),
    card_id="backtest-replay-ex-post",
):
    policy_picker, _ = st.columns([1.2, 4.8])
    with policy_picker:
        replay_policy = st.selectbox(
            "Replay policy",
            options=POLICY_ORDER,
            index=POLICY_ORDER.index("Dynamic"),
            format_func=lambda policy: policy.split(" · ")[0],
            key=f"backtest_replay_policy_{market_zone}",
        )
    replay_policy_short = replay_policy.split(" · ")[0]
    replay_ex_post = settlement.loc[
        (settlement["delivery_start_local"].dt.date == replay_day)
        & (settlement["policy_name"] == POLICY_SETTLEMENT_NAMES[replay_policy])
    ].copy()
    if replay_ex_post.empty:
        st.info("No settlement replay data is available for the selected policy and day.")
        st.stop()
    replay_plot = replay_ex_post.copy()
    replay_plot["imbalance_direction"] = np.where(
        replay_plot["portfolio_actual_mwh"] >= replay_plot["nomination_mwh"],
        "Surplus",
        "Shortfall",
    )
    replay_plot["delivery_end_local"] = replay_plot[
        "delivery_start_local"
    ] + pd.Timedelta(minutes=15)
    line_data = pd.concat(
        [
            replay_plot.assign(
                series="Actual production",
                energy_mwh=replay_plot["portfolio_actual_mwh"],
            ),
            replay_plot.assign(
                series=f"{replay_policy.split(' · ')[0]} nomination",
                energy_mwh=replay_plot["nomination_mwh"],
            ),
        ],
        ignore_index=True,
    )
    imbalance_area = alt.Chart(replay_plot).mark_rect(opacity=0.18).encode(
        x=alt.X("delivery_start_local:T", title="Delivery time"),
        x2="delivery_end_local:T",
        y=alt.Y(
            "portfolio_actual_mwh:Q",
            title="Energy per 15-min interval (MWh)",
        ),
        y2="nomination_mwh:Q",
        color=alt.Color(
            "imbalance_direction:N",
            title="Position",
            scale=alt.Scale(
                domain=["Surplus", "Shortfall"],
                range=["#6FAF8E", "#C87878"],
            ),
        ),
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("imbalance_direction:N", title="Position"),
            alt.Tooltip("portfolio_actual_mwh:Q", title="Actual production (MWh)", format=",.2f"),
            alt.Tooltip(
                "nomination_mwh:Q",
                title=f"{replay_policy_short} nomination (MWh)",
                format=",.2f",
            ),
        ],
    )
    lines = alt.Chart(line_data).mark_line(
        interpolate="step-after",
        strokeWidth=2.5,
    ).encode(
        x=alt.X("delivery_start_local:T", title="Delivery time"),
        y=alt.Y(
            "energy_mwh:Q",
            title="Energy per 15-min interval (MWh)",
        ),
        color=alt.Color(
            "series:N",
            title="Series",
            sort=["Actual production", f"{replay_policy_short} nomination"],
            scale=alt.Scale(
                domain=["Actual production", f"{replay_policy_short} nomination"],
                range=["#FFFFFF", "#D9B36C"],
            ),
        ),
        tooltip=[
            alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("energy_mwh:Q", title="Energy (MWh)", format=",.2f"),
            alt.Tooltip("imbalance_volume_mwh:Q", title="Signed imbalance (MWh)", format=",.2f"),
            alt.Tooltip("gross_pnl_eur:Q", title="Gross PnL (€)", format=",.2f"),
        ],
    )
    ex_post_chart = (
        alt.layer(imbalance_area, lines)
        .resolve_scale(color="independent")
        .properties(height=300)
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
    st.altair_chart(ex_post_chart, use_container_width=True)

    price_data = pd.concat(
        [
            replay_plot.assign(
                series="Realised DA Price",
                price_eur_mwh=replay_plot["actual_day_ahead_price_eur_mwh"],
            ),
            replay_plot.assign(
                series="Applied reBAP",
                price_eur_mwh=replay_plot["applied_rebap_eur_mwh"],
            ),
        ],
        ignore_index=True,
    )
    price_chart = (
        alt.Chart(price_data)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X("delivery_start_local:T", title="Delivery time"),
            y=alt.Y("price_eur_mwh:Q", title="Price (€/MWh)"),
            color=alt.Color(
                "series:N",
                title="Series",
                sort=["Realised DA Price", "Applied reBAP"],
                scale=alt.Scale(
                    domain=["Realised DA Price", "Applied reBAP"],
                    range=["#4F8FCB", "#D9B36C"],
                ),
            ),
            tooltip=[
                alt.Tooltip("delivery_start_local:T", title="Delivery time", format="%H:%M"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("price_eur_mwh:Q", title="Price (€/MWh)", format=",.2f"),
            ],
        )
        .properties(height=220)
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
    st.caption("Realised market prices")
    st.altair_chart(price_chart, use_container_width=True)

    daily_summary = {
        "DA Revenue": replay_ex_post["day_ahead_revenue_eur"].sum(),
        "Imbalance Settlement": replay_ex_post["imbalance_settlement_eur"].sum(),
        "Gross PnL": replay_ex_post["gross_pnl_eur"].sum(),
        "Shortfall": (-replay_ex_post["imbalance_volume_mwh"]).clip(lower=0).sum(),
        "Surplus": replay_ex_post["imbalance_volume_mwh"].clip(lower=0).sum(),
        "Abs. Imbalance": replay_ex_post["imbalance_volume_mwh"].abs().sum(),
    }
    st.caption("Delivery-day economic summary")
    summary_columns = st.columns(6)
    for column, (label, value) in zip(summary_columns, daily_summary.items()):
        unit = "€" if label in {"DA Revenue", "Imbalance Settlement", "Gross PnL"} else "MWh"
        column.metric(label, f"{value:,.0f} {unit}")

    ex_post_table = replay_ex_post[
        [
            "delivery_start_local",
            "portfolio_actual_mwh",
            "nomination_mwh",
            "actual_day_ahead_price_eur_mwh",
            "applied_rebap_eur_mwh",
            "imbalance_volume_mwh",
            "imbalance_settlement_eur",
            "gross_pnl_eur",
        ]
    ].rename(
        columns={
            "delivery_start_local": "Delivery time",
            "portfolio_actual_mwh": "Actual production (MWh)",
            "nomination_mwh": f"{replay_policy_short} nomination (MWh)",
            "actual_day_ahead_price_eur_mwh": "DA price (€/MWh)",
            "applied_rebap_eur_mwh": "Applied reBAP (€/MWh)",
            "imbalance_volume_mwh": "Signed imbalance (MWh)",
            "imbalance_settlement_eur": "Imbalance Settlement (€)",
            "gross_pnl_eur": "Gross PnL (€)",
        }
    )
    with st.expander("Show EX-POST quarter-hour details"):
        st.dataframe(
            ex_post_table,
            hide_index=True,
            width="stretch",
        )
