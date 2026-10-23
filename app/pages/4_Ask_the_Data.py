"""Plain-English questions -> one guarded, read-only SQL query -> table -> short summary."""

import os

import plotly.express as px
import streamlit as st
import ui

from couponroi import ask as A

ui.page("Ask the Data", "💬")
ui.bundle_ready()

st.title("Ask the data")
st.markdown(
    "Type a question. An LLM writes **one** SQL query, which is checked before it runs: a single SELECT, only "
    "the documented tables, no file or URL access, and a row limit. The database is opened read-only with external "
    "access switched off, so even a bad query can't change or leak anything."
)


def llm_settings() -> dict | None:
    secrets = {}
    try:
        secrets = dict(st.secrets.get("llm", {}))
    except Exception:  # no secrets file
        secrets = {}
    base = secrets.get("base_url") or os.environ.get("COUPONROI_LLM_BASE_URL")
    if not base:
        return None
    return {
        "base_url": base,
        "model": secrets.get("model") or os.environ.get("COUPONROI_LLM_MODEL"),
        "api_key": secrets.get("api_key") or os.environ.get("COUPONROI_LLM_API_KEY"),
    }


examples = A.load_eval()
settings = llm_settings()
con = ui.db().cursor()


def show(df, sql, summary=None):
    st.code(sql, language="sql")
    if summary:
        st.info(summary)
    st.dataframe(df, hide_index=True, width="stretch")
    if df.shape[1] == 2 and len(df) > 1 and df.dtypes.iloc[1].kind in "if":
        x, y = df.columns
        fig = (px.line if df[x].dtype.kind in "iufM" and len(df) > 12 else px.bar)(df, x=x, y=y)
        fig.update_traces(marker_color=ui.BLUE) if len(df) <= 12 else fig.update_traces(line_color=ui.BLUE)
        fig.update_layout(height=320, title=None)
        st.plotly_chart(fig, width="stretch")


if settings is None:
    st.warning(
        "No LLM is configured for this deployment, so free-text questions are off. Pick an example below: it runs the "
        "reference SQL from the evaluation set through the same guardrails.\n\n"
        "To turn it on, set `COUPONROI_LLM_BASE_URL` (and `COUPONROI_LLM_MODEL`, `COUPONROI_LLM_API_KEY` if needed), "
        "or add an `[llm]` block to `.streamlit/secrets.toml`. Any OpenAI-compatible endpoint works: Ollama locally "
        "(free), Groq or Gemini (free tiers)."
    )
    q = st.selectbox("Example question", [e["question"] for e in examples])
    item = next(e for e in examples if e["question"] == q)
    sql = A.guard(item["sql"])
    show(con.execute(sql).df(), sql)
else:
    client = A.LLMClient(A.LLMConfig.from_env(settings))
    st.caption(f"Model: `{client.cfg.model}` at `{client.cfg.base_url}`")
    cols = st.columns(3)
    for i, e in enumerate(examples[:6]):
        if cols[i % 3].button(e["question"], key=f"ex{i}", width="stretch"):
            st.session_state["question"] = e["question"]
    question = st.text_input(
        "Your question", key="question", placeholder="e.g. Which departments grew most in the second half?"
    )
    if question:
        with st.spinner("Writing and checking the query..."):
            res = A.ask(question, con, client)
        if res.error:
            st.error(res.error)
            if res.raw:
                with st.expander("What the model returned"):
                    st.code(res.raw)
        else:
            show(res.data, res.sql, res.summary)

with st.expander("Tables the model can see"):
    st.text(A.schema_text())
ui.footer()
