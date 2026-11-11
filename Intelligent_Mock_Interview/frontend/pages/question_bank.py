"""
Question Bank Management Page
=============================
Explore, filter, and add categorized interview questions with benchmark
reference answers and key expected concepts.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, empty_state
from frontend.services.api_client import client


def render_question_bank(nav_to_fn):
    page_header(
        kicker="Curriculum Management",
        title="Question Bank Explorer",
        description="Inspect, filter, and augment domain-specific technical interview questions paired with expert reference answers and expected concept tags.",
    )

    t_explore, t_add = st.tabs(["📚 Question Explorer", "➕ Add New Question"])

    # ── TAB 1: QUESTION EXPLORER ─────────────────────────────────────────────
    with t_explore:
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            domains = ["All Domains"] + client.get_domains()
            domain_filter = st.selectbox("Filter by Domain:", domains, index=0)
        with c2:
            difficulty_filter = st.selectbox("Filter by Difficulty:", ["All Difficulties", "Beginner", "Intermediate", "Advanced"], index=0)
        with c3:
            search_query = st.text_input("Search questions:", placeholder="Search keywords…")

        questions = client.get_questions(domain=domain_filter, difficulty=difficulty_filter, search=search_query, limit=100)

        st.write("")
        st.markdown(f"**Found {len(questions)} question(s)**")

        if questions:
            for q in questions:
                with st.container(border=True):
                    q_col1, q_col2 = st.columns([3, 1])
                    with q_col1:
                        st.markdown(f"**[{q.get('question_code')}]** · `{q.get('domain')}` · *{q.get('topic')}*")
                        st.markdown(f"#### {q.get('question_text')}")
                    with q_col2:
                        diff = q.get("difficulty", "Intermediate")
                        st.markdown(f"**Difficulty:** `{diff}`")
                        st.caption(f"Source: {q.get('provenance') or 'Curated'}")

                    concepts = q.get("expected_concepts", [])
                    if concepts:
                        st.markdown("**Expected Concepts:** " + " · ".join(f"`{c}`" for c in concepts))

                    if q.get("reference_answer"):
                        with st.expander("View Benchmark Reference Answer"):
                            st.write(q.get("reference_answer"))
        else:
            empty_state("📖", "No Questions Found", "Try broadening your filters or add a new question to the bank.")

    # ── TAB 2: ADD NEW QUESTION ──────────────────────────────────────────────
    with t_add:
        section_header("Add Question to Repository", "Contribute a new question with reference answer and concepts")
        with st.form("form_add_question", clear_on_submit=True):
            f1, f2 = st.columns(2)
            with f1:
                q_domain = st.selectbox("Domain *", client.get_domains())
                q_diff = st.selectbox("Difficulty *", ["Beginner", "Intermediate", "Advanced"], index=1)
            with f2:
                q_topic = st.text_input("Topic / Sub-area", placeholder="e.g. Memory Management, Trees, Normalization")
                q_provenance = st.text_input("Provenance / Citation", placeholder="e.g. Standard University Syllabus")

            q_text_input = st.text_area("Question Text *", height=100, placeholder="Type the interview question clearly…")
            q_ref_input = st.text_area("Benchmark Reference Answer *", height=140, placeholder="Provide the ideal technical benchmark explanation…")
            q_concepts_input = st.text_input("Expected Concepts (comma-separated)", placeholder="e.g. Polymorphism, Virtual Table, Runtime Binding")

            submitted = st.form_submit_button("Add Question to Bank", type="primary", use_container_width=True)

            if submitted:
                if not q_text_input.strip() or not q_ref_input.strip():
                    st.error("Question text and benchmark reference answer are required.")
                else:
                    concepts_list = [c.strip() for c in q_concepts_input.split(",") if c.strip()]
                    payload = {
                        "domain": q_domain,
                        "topic": q_topic.strip() or "General",
                        "difficulty": q_diff,
                        "question_text": q_text_input.strip(),
                        "reference_answer": q_ref_input.strip(),
                        "expected_concepts": concepts_list,
                        "provenance": q_provenance.strip() or "Added by InterviewAI Administrator",
                    }
                    success, res = client.create_question(payload)
                    if success:
                        st.success(f"Question added successfully! Code: `{res.get('question_code')}`")
                    else:
                        st.error(f"Failed to add question: {res}")

