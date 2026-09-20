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
            --sidebar-block-spacing: 0.65rem;
            --sidebar-block-spacing-wide: 0.85rem;
        }

        [data-testid="stAppViewContainer"] { background: #000000; }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background: #151E2A;
            border-right: 1px solid #29394B;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }

        [data-testid="stSidebar"] .st-key-sidebar-zone-block,
        [data-testid="stSidebar"] .st-key-sidebar-portfolio-note,
        [data-testid="stSidebar"] .st-key-sidebar-period-block,
        [data-testid="stSidebar"] .st-key-sidebar-period-meta,
        [data-testid="stSidebar"] .st-key-sidebar-modules-heading {
            margin-bottom: var(--sidebar-block-spacing);
        }

        [data-testid="stSidebar"] .st-key-sidebar-title-block,
        [data-testid="stSidebar"] .st-key-sidebar-portfolio-note,
        [data-testid="stSidebar"] .st-key-sidebar-period-meta {
            margin-bottom: var(--sidebar-block-spacing-wide);
        }

        .sidebar-app-title {
            margin: 0;
            color: #E7EBF0;
            font-size: 1.18rem;
            font-weight: 650;
            line-height: 1.25;
        }

        .sidebar-context-note {
            margin: 0 !important;
            color: #93A2B5;
            font-size: 0.72rem;
            line-height: 1.3;
        }

        .sidebar-section-label {
            display: block;
            margin: 0;
            color: #93A6BA;
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.12em;
        }

        [data-testid="stSidebar"] .st-key-sidebar-modules-heading {
            position: relative;
            z-index: 2;
            padding: 10px 0 2px 2px;
            border-top: 1px solid #29394B;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stDateInput"] label {
            min-height: 0;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
        [data-testid="stSidebar"] [data-testid="stDateInput"] label p {
            font-size: 0.76rem;
        }

        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0.92); }

        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            box-sizing: border-box;
            min-height: 1.85rem;
            margin: 1px 3px 1px 0 !important;
            padding: 4px 9px !important;
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
            background: #20352D !important;
            border-color: transparent !important;
            border-left: 3px solid #6FAF8E !important;
            padding-left: 7px !important;
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
            min-height: 1.85rem;
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
            background: #18232F;
            border-left: 2px solid transparent;
            color: #C5CFDB;
            font-weight: 600;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary * {
            color: #C5CFDB !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #9EADBF;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary svg {
            color: #9EADBF !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
            padding-left: 16px;
        }

        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {
            gap: 0.2rem;
        }

        [data-testid="stMetric"] {
            background: var(--dashboard-card-background) !important;
            border: 1px solid var(--dashboard-card-border) !important;
            border-radius: 9px;
            padding: 12px;
        }

        .st-key-production-secondary-metrics [data-testid="stMetric"],
        .st-key-price-secondary-metrics [data-testid="stMetric"] {
            background: #111923 !important;
            border-color: #304A40 !important;
            padding: 9px 11px;
        }

        .st-key-production-secondary-metrics [data-testid="stMetricLabel"] p,
        .st-key-price-secondary-metrics [data-testid="stMetricLabel"] p {
            color: #9EADBF !important;
            font-size: 0.78rem;
        }

        .st-key-production-secondary-metrics [data-testid="stMetricValue"],
        .st-key-price-secondary-metrics [data-testid="stMetricValue"] {
            font-size: 1.35rem;
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
            min-height: 48px;
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
            min-height: 48px;
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
            min-height: 40px !important;
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

        [class*="st-key-nomination-p50-alert"] [data-testid="stAlert"] {
            width: min(100%, 700px);
            max-width: 100%;
            margin: 0;
            min-height: 0;
            padding: 0.4rem 0.7rem;
            display: flex;
            align-items: center;
            background: #2B2A22;
            border: 1px solid #5A5034;
            border-left: 3px solid #D9B36C;
            color: #E7D8B5;
        }

        [class*="st-key-nomination-p50-alert"] [data-testid="stAlert"] * {
            color: #E7D8B5 !important;
        }

        [class*="st-key-nomination-p50-alert"] [data-testid="stAlert"] .stAlertContainer {
            background: #2B2A22 !important;
        }

        [class*="st-key-nomination-p50-alert"] [data-testid="stAlert"] p {
            margin: 0;
            text-align: left;
        }

        [class*="st-key-nomination-p50-alert"] {
            width: 100%;
            max-width: 100%;
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
    with st.container(key="sidebar-title-block"):
        st.markdown(
            '<div class="sidebar-app-title">Renewable Portfolio</div>',
            unsafe_allow_html=True,
        )
    with st.container(key="sidebar-zone-block"):
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
    with st.container(key="sidebar-portfolio-note"):
        st.markdown(
            '<p class="sidebar-context-note">'
            + (
                "Virtual Qair portfolio"
                if zone_config.get("available", True)
                else "No data available for this bidding zone"
            )
            + "</p>",
            unsafe_allow_html=True,
        )
    if zone_config.get("available", True):
        # The zone and delivery period define the context for every page below.
        try:
            first_day, last_day = available_period(market_zone)
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Unable to load the backtest period: {error}")
            st.stop()

        with st.container(key="sidebar-period-block"):
            period = st.date_input(
                "Delivery period",
                value=(first_day, last_day),
                min_value=first_day,
                max_value=last_day,
                format="DD-MM-YYYY",
                key=f"delivery_period_{market_zone}",
            )
        st.session_state["delivery_period"] = period
        with st.container(key="sidebar-period-meta"):
            st.markdown(
                '<p class="sidebar-context-note">'
                f"Historical backtest · {zone_config['timezone']} · "
                f"{zone_config['resolution_minutes']}-min"
                "</p>",
                unsafe_allow_html=True,
            )
    else:
        period = ()
    with st.container(key="sidebar-modules-heading"):
        st.markdown(
            '<span class="sidebar-section-label">MODULES</span>',
            unsafe_allow_html=True,
        )
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

if not zone_config.get("available", True):
    navigation.run()
    st.stop()

if len(period) != 2:
    st.info("Select an end date to complete the delivery period.")
    st.stop()

capacity_mw = sum(asset["capacity_mw"] for asset in zone_config["assets"])
st.caption(
    f"{market_zone} · {zone_config['portfolio_name']} · "
    f"{capacity_mw:.1f} MW · {zone_config['resolution_minutes']}-min"
)
navigation.run()
