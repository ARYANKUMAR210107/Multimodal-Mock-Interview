"""
InterviewAI Reusable UI Components
==================================
Design system building blocks for consistent layouts, typography,
KPI cards, status badges, and empty states.
"""
from __future__ import annotations

import streamlit as st
from typing import Callable, Optional


def page_header(kicker: str, title: str, description: str = ""):
    """Render the standard top-level page header with modern typography."""
    st.markdown(
        f"""
        <div class="page-header-wrapper">
            <div class="page-eyebrow">{kicker}</div>
            <h1 class="page-title">{title}</h1>
            {f'<p class="page-description">{description}</p>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    """Render a clean card section header."""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, subtext: str = "", accent: str = "purple"):
    """Render a styled KPI metric card with micro-border accent."""
    accent_class = ""
    if accent in ("cyan", "blue"): accent_class = "blue"
    elif accent in ("green", "emerald"): accent_class = "green"
    elif accent in ("amber", "yellow", "orange"): accent_class = "amber"

    st.markdown(
        f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-header">
                <span class="kpi-label">{label}</span>
            </div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(status: str, label: Optional[str] = None):
    """Render an inline status badge (online, warning, offline)."""
    st_norm = status.lower()
    css_class = "online" if st_norm in ("online", "active", "completed", "assessed", "success", "true") else (
        "warning" if st_norm in ("warning", "in_progress", "pending", "evaluating") else "offline"
    )
    display_label = label or status.capitalize()
    st.markdown(
        f"""
        <span class="status-pill {css_class}">
            <span class="status-dot"></span>
            {display_label}
        </span>
        """,
        unsafe_allow_html=True,
    )


def tier_badge(score: float) -> str:
    """Calculate and return HTML for a performance diagnostic tier."""
    if score >= 0.70:
        cls_name = "proficient"
        label = "Proficient / Strong Answer"
    elif score >= 0.45:
        cls_name = "competent"
        label = "Competent Coverage"
    else:
        cls_name = "developing"
        label = "Developing / Practice Needed"

    return f'<span class="tier-badge {cls_name}">{label}</span>'


def empty_state(icon: str, title: str, copy: str, action_label: str = "", on_click: Optional[Callable] = None, key: Optional[str] = None):
    """Render a standardized empty state banner with optional call-to-action."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and on_click:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.button(action_label, key=key or f"btn_empty_{title[:10]}", type="primary", use_container_width=True, on_click=on_click)

