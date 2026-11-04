import streamlit as st
from agent import run_agent

from observability.trace_store import init_db, list_runs, get_run

init_db()

st.set_page_config(page_title="Multi-Agent Freshness Research", page_icon="🤖", layout="wide")
st.title("Multi-Agent Research Agent")
st.caption("Research | Freshness classification | Evidence verification | Refresh | Trace")

query = st.text_area(
    "Ask the agent",
    placeholder="What is the latest version of Go (Golang)?",
    height=100,
)

if st.button("Run Agent", type="primary"):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        with st.spinner("Running agents and checking freshness..."):
            try:
                st.session_state["result"] = run_agent(query.strip())
            except Exception as exc:
                st.error(f"Agent execution failed: {exc}")

if "result" in st.session_state:
    result = st.session_state["result"]
    print(result)
    st.divider()
    st.subheader("Final Answer")
    st.markdown(result.get("answer", "No answer returned."))
    fresh = result.get("freshness") or {}
    st.caption(
        f"Trace ID: {result.get('trace_id')} | "
        f"Time-sensitive: {fresh.get('time_sensitive')} | "
        f"Verification: {fresh.get('verification_status')}"
    )

    if result.get("was_refreshed"):
        st.subheader("Original vs Refreshed Answer")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original")
            st.warning("Before evidence-based refresh")
            st.markdown(result.get("original_answer", ""))
            st.json(result.get("original_freshness", {}))
        with col2:
            st.markdown("#### Refreshed")
            st.success("Regenerated using retrieved evidence")
            st.markdown(result.get("answer", ""))
            st.json(result.get("verification") or {})
    else:
        st.info("No refresh was required.")
        with st.expander("Original answer"):
            st.markdown(result.get("original_answer", ""))

    with st.expander("Evidence sources"):
        st.json(result.get("evidence", {}))

    with st.expander("Agent execution trace"):
        for item in (result.get("trace") or {}).get("spans", []):
            label = (f"{item.get('agent_name')} | {item.get('status')} | "
                     f"{item.get('duration_ms')} ms")
            with st.expander(label):
                st.write("Span ID:", item.get("span_id"))
                st.write("Parent ID:", item.get("parent_id"))
                st.write("Started:", item.get("started_at"))
                st.write("Input:")
                st.json(item.get("input"))
                st.write("Output:")
                st.json(item.get("output"))
                if item.get("error"):
                    st.error(item["error"])
    
    st.divider()
    st.subheader("Saved run history")
    try:
        history = list_runs(limit=30)
        if history:
            labels = [f"{row['created_at']} 路 {row['query'][:100]}" for row in history]
            selected = st.selectbox("Previous runs", labels)
            chosen = history[labels.index(selected)]
            if st.button("Load selected run"):
                st.session_state["history_run"] = get_run(chosen["trace_id"])
            if st.session_state.get("history_run"):
                st.json(st.session_state["history_run"])
        else:
            st.caption("No saved runs yet.")
    except Exception as exc:
        st.warning(f"Could not load saved history: {exc}")