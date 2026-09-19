"""Entry point for the portfolio backtest dashboard."""

import streamlit as st

from src.dashboard.data import (
    DEFAULT_MARKET_ZONE,
    MARKET_ZONES,
    available_period,
    market_zone_config,
)

st.set_page_config(page_title="Renewable Portfolio", layout="wide")

st.markdown(
    """
    <style>
        /* A restrained dark palette, close to an internal energy-trading tool. */
        :root {
            --dashboard-card-background: #151E2A;
            --dashboard-card-border: #6FAF8E;
            --dashboard-card-gutter: 14px;
        }

        [data-testid="stAppViewContainer"] { background: #000000; }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background: #151E2A;
            border-right: 1px solid #29394B;
        }

        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0.92); }

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            box-sizing: border-box;
            min-height: 2rem;
            margin: 2px 0 !important;
            padding: 5px 10px !important;
            border: 1px solid transparent !important;
            border-radius: 7px !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:link,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:visited,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:active,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *::before,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *::after {
            color: #B4C0CF !important;
            -webkit-text-fill-color: #B4C0CF !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: #1B2838 !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover * {
            color: #E7EBF0 !important;
            -webkit-text-fill-color: #E7EBF0 !important;
        }

        [data-testid="stSidebar"] .st-key-sidebar-active-page-link [data-testid="stPageLink-NavLink"] {
            background: #1A302B !important;
            border-color: #36584C !important;
        }

        [data-testid="stSidebar"] .st-key-sidebar-active-page-link [data-testid="stPageLink-NavLink"],
        [data-testid="stSidebar"] .st-key-sidebar-active-page-link [data-testid="stPageLink-NavLink"] * {
            color: #D7EDE1 !important;
            -webkit-text-fill-color: #D7EDE1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border: 1px solid transparent;
            border-radius: 7px;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 2rem;
            box-sizing: border-box;
            border-radius: 7px;
            color: #B4C0CF;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: #B4C0CF !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background: #1B2838;
            color: #E7EBF0;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover * {
            color: #E7EBF0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
            background: #1A302B;
            color: #D7EDE1;
            font-weight: 600;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary * {
            color: #D7EDE1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #9EADBF;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary svg {
            color: #6FAF8E !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
            padding-left: 12px;
        }

        [data-testid="stMetric"] {
            background: var(--dashboard-card-background) !important;
            border: 1px solid var(--dashboard-card-border) !important;
            border-radius: 9px;
            padding: 12px;
        }

        [class*="st-key-chart-card-"] {
            box-sizing: border-box;
            background: var(--dashboard-card-background) !important;
            border: 1px solid var(--dashboard-card-border) !important;
            border-radius: 9px !important;
            padding: var(--dashboard-card-gutter) !important;
        }

        .chart-card-heading {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .st-key-chart-card-nomination-hourly-choice-frequency .chart-card-heading,
        .st-key-chart-card-nomination-hourly-scores .chart-card-heading {
            min-height: 42px;
        }

        .st-key-chart-card-nomination-hourly-choice-frequency .chart-card-description,
        .st-key-chart-card-nomination-hourly-scores .chart-card-description {
            min-height: 15px;
        }

        .st-key-chart-card-nomination-hourly-choice-frequency,
        .st-key-chart-card-nomination-hourly-scores {
            display: flex;
            flex-direction: column;
            height: 540px !important;
            min-height: 540px !important;
        }

        .chart-card-title {
            color: #E7EBF0;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.25;
        }

        .chart-card-description {
            color: #9EADBF;
            font-size: 0.75rem;
            line-height: 1.25;
        }

        .chart-card-divider {
            height: 1px;
            width: calc(100% + 2 * var(--dashboard-card-gutter));
            margin: 5px calc(-1 * var(--dashboard-card-gutter)) 0;
            background: var(--dashboard-card-border);
        }

        [class*="st-key-chart-card-"] [data-testid="stPopoverButton"] {
            min-height: 1.5rem;
            padding: 0.1rem;
            border: 0;
            background: transparent;
            color: #9EADBF;
        }

        [class*="st-key-chart-card-"] [data-testid="stPopoverButton"]:hover {
            color: var(--dashboard-card-border);
        }

        [data-testid="stButton"] button {
            min-height: 56px;
            border: 1px solid #3D634D;
            border-radius: 9px;
            background: #151E2A;
            color: #D9E4DC;
        }

        [data-testid="stButton"] button:hover {
            border-color: #6FAF8E;
            color: #E7EBF0;
        }

        [data-testid="stButton"] button[kind="primary"] {
            border-color: #6FAF8E;
            background: #1A302B;
            color: #E3F0E8;
        }

        [class*="st-key-policy-card-dynamic-"] {
            box-sizing: border-box;
            display: flex;
            min-height: 56px;
            align-items: center;
            border: 1px solid #3D634D !important;
            border-radius: 9px !important;
            background: #151E2A !important;
            padding: 2px 4px !important;
        }

        [class*="st-key-policy-card-dynamic-selected"] {
            border-color: #6FAF8E !important;
            background: #1A302B !important;
        }

        [class*="st-key-policy-card-dynamic-"] [data-testid="stButton"] button {
            min-height: 48px !important;
            border: 0 !important;
            background: transparent !important;
            color: #D9E4DC !important;
            padding: 0.25rem !important;
        }

        [class*="st-key-policy-card-dynamic-"] [data-testid="stPopoverButton"] {
            width: 24px !important;
            min-height: 32px !important;
            border: 0 !important;
            background: transparent !important;
            color: #9EADBF !important;
            padding: 0 !important;
        }

        [class*="st-key-policy-card-dynamic-"] [data-testid="stPopoverButton"]:hover {
            color: #6FAF8E !important;
        }

        [data-testid="stDateInput"] input {
            background: #182230;
            border-color: #34465C;
            color: #E7EBF0;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #151E2A;
            border-color: #3D634D;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
            border-color: #6FAF8E;
            box-shadow: 0 0 0 1px #6FAF8E;
        }

        h1, h2, h3 { color: #E7EBF0; letter-spacing: -0.01em; }

        [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] .stCaption { color: #9EADBF; }

        hr { border-color: #29394B; }
    </style>
    """,
    unsafe_allow_html=True,
)

portfolio_page = st.Page("pages/portfolio.py", title="Portfolio Overview", default=True)
production_page = st.Page("pages/production.py", title="Production Forecast")
price_page = st.Page("pages/price.py", title="Day-Ahead Price")
nomination_page = st.Page(
    "pages/nomination_strategy.py", title="Nomination Strategy"
)
backtest_page = st.Page("pages/backtest.py", title="Backtest & Settlement")

navigation = st.navigation(
    [portfolio_page, production_page, price_page, nomination_page, backtest_page],
    position="hidden",
)


def sidebar_page_link(page, label: str) -> None:
    """Render a page link and mark the current page with the green accent."""
    if navigation.url_path == page.url_path:
        with st.container(key="sidebar-active-page-link"):
            st.page_link(page, label=label)
    else:
        st.page_link(page, label=label)

with st.sidebar:
    st.title("Renewable Portfolio")
    market_zone = st.selectbox(
        "Bidding zone",
        options=list(MARKET_ZONES),
        index=list(MARKET_ZONES).index(DEFAULT_MARKET_ZONE),
        format_func=lambda zone_id: (
            f"{zone_id} · {MARKET_ZONES[zone_id]['display_name']}"
        ),
        key="market_zone",
        disabled=len(MARKET_ZONES) == 1,
    )
    zone_config = market_zone_config(market_zone)
    asset_names = [asset["name"] for asset in zone_config["assets"]]
    listed_assets = (
        " and ".join(asset_names)
        if len(asset_names) == 2
        else ", ".join(asset_names)
    )
    verb = "is" if len(asset_names) == 1 else "are"
    st.caption(
        f"{listed_assets} {verb} virtual assets used for this backtest, "
        "not Qair's real operating portfolio."
    )
    # The zone and delivery period define the context for every page below.
    try:
        first_day, last_day = available_period(market_zone)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Unable to load the backtest period: {error}")
        st.stop()

    period = st.date_input(
        "Delivery period",
        value=(first_day, last_day),
        min_value=first_day,
        max_value=last_day,
        format="DD/MM/YYYY",
        key=f"delivery_period_{market_zone}",
    )
    st.session_state["delivery_period"] = period
    st.caption(
        f"Delivery times: {zone_config['timezone']} Â· "
        f"{zone_config['resolution_minutes']}-minute resolution"
    )
    st.divider()
    sidebar_page_link(portfolio_page, "Portfolio Overview")
    sidebar_page_link(production_page, "Production Forecast")
    sidebar_page_link(price_page, "Day-Ahead Price")
    strategy_pages = {nomination_page.url_path, backtest_page.url_path}
    with st.expander(
        "Strategy & Backtest",
        expanded=navigation.url_path in strategy_pages,
    ):
        sidebar_page_link(nomination_page, "Nomination Strategy")
        sidebar_page_link(backtest_page, "Backtest & Settlement")

    st.caption("Historical backtest · Virtual production")

if len(period) != 2:
    st.info("Select an end date to complete the delivery period.")
    st.stop()

capacity_mw = sum(asset["capacity_mw"] for asset in zone_config["assets"])
st.caption(
    f"{market_zone} · {zone_config['portfolio_name']} · "
    f"{capacity_mw:.1f} MW · {zone_config['resolution_minutes']}-min"
)
navigation.run()
