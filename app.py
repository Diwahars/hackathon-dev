import json
import os
import sqlite3

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

DB_PATH = os.getenv(
    "DATABASE_PATH",
    "agent_traces.db",
)

st.set_page_config(
    page_title="Agent Observability",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Agent Observability")
st.caption(
    "Bifrost Gateway + Tavily Freshness Detection"
)


def load_traces():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                trace_id,
                query,
                answer,
                status,
                started_at,
                duration_ms,
                events
            FROM traces
            ORDER BY started_at DESC
            """,
            conn,
        )


query = st.text_area(
    "Ask the agent",
    placeholder=(
        "What is the latest version of Python?"
    ),
)

if st.button("Run Agent", type="primary"):
    if not query.strip():
        st.warning("Enter a query first.")

    else:
        with st.spinner("Executing agent..."):
            try:
                result = run_agent(query)

                st.session_state["result"] = result

            except Exception as exc:
                st.error(str(exc))


result = st.session_state.get("result")

if result:
    st.subheader("Final Answer")

    st.write(result["answer"])

    freshness = result.get("freshness", {})

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Freshness",
        "Stale" if freshness.get("stale") else "Fresh",
    )

    col2.metric(
        "Stale Score",
        freshness.get("score", 0),
    )

    col3.metric(
        "Trace ID",
        result["trace_id"][:8],
    )

    st.subheader("Tool Execution Timeline")

    events = result["trace"]["events"]

    tool_events = [
        event for event in events
        if event["event_type"] == "tool_completed"
    ]

    if tool_events:
        df = pd.DataFrame(tool_events)

        st.dataframe(
            df[
                [
                    "tool_name",
                    "status",
                    "duration_ms",
                ]
            ],
            use_container_width=True,
        )

        st.bar_chart(
            df.set_index("tool_name")[
                "duration_ms"
            ]
        )

    st.subheader("Freshness Verification")

    for item in freshness.get("results", []):
        with st.expander(
            item.get("claim", "Claim")
        ):
            st.write(
                "Status:",
                item.get("status"),
            )

            st.write(
                "Confidence:",
                item.get("confidence"),
            )

            st.write(
                "Explanation:",
                item.get("explanation"),
            )

    with st.expander("Raw Trace Events"):
        st.json(events)


st.divider()

st.subheader("Historical Agent Traces")

history = load_traces()

if not history.empty:
    st.dataframe(
        history[
            [
                "trace_id",
                "query",
                "status",
                "duration_ms",
            ]
        ],
        use_container_width=True,
    )
else:
    st.info("No traces available yet.")