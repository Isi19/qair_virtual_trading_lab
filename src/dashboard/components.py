"""Reusable visual components for the Streamlit dashboard."""

from collections.abc import Iterator
from contextlib import contextmanager
from html import escape

import streamlit as st


@contextmanager
def chart_card(
    title: str,
    description: str,
    card_id: str,
    info_text: str | None = None,
) -> Iterator[None]:
    """Show a chart inside the dashboard's standard framed card."""
    with st.container(
        border=True,
        key=f"chart-card-{card_id}",
        gap="small",
    ):
        if info_text:
            title_column, info_column = st.columns(
                [0.94, 0.06],
                gap="small",
                vertical_alignment="top",
            )
            with title_column:
                _show_chart_heading(title, description)
            with info_column:
                with st.popover(
                    "ⓘ",
                    help="More information",
                ):
                    st.write(info_text)
        else:
            _show_chart_heading(title, description)

        st.markdown(
            '<div class="chart-card-divider"></div>',
            unsafe_allow_html=True,
        )
        yield


def _show_chart_heading(title: str, description: str) -> None:
    """Render the title and description shared by chart cards."""
    st.markdown(
        f'''
        <div class="chart-card-heading">
            <div class="chart-card-title">{escape(title)}</div>
            <div class="chart-card-description">{escape(description)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
