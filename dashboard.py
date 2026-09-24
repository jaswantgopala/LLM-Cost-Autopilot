"""
Phase 4: Cost Dashboard — Streamlit app.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.logger import get_all_requests, get_summary_stats, init_db
from core.registry import MODEL_REGISTRY

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")

init_db()

st.title("🎯 LLM Cost Autopilot Dashboard")
st.caption("Intelligent request routing to minimize LLM API costs while maintaining quality")

# --- Load data ---
requests = get_all_requests()
stats = get_summary_stats()

if not requests:
    st.warning("No requests logged yet. Run some prompts through the pipeline first (`python scripts/test_pipeline.py`).")
    st.stop()

df = pd.DataFrame(requests)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- The Money Shot: Cost Reduction Percentage ---
# Reference baseline: GPT-4o published pricing (industry-standard premium model).
# NOTE: GPT-4o is NOT called anywhere in this project — this is a reference-only
# cost calculation using GPT-4o's public per-token pricing, applied to the same
# token counts your routed requests actually used. This lets us show a realistic
# "what if every request went to a premium flagship model" comparison at zero cost.
REFERENCE_BASELINE_NAME = "GPT-4o (reference only, not called)"
REFERENCE_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000   # published OpenAI pricing
REFERENCE_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000

baseline_cost = 0.0
for _, row in df.iterrows():
    baseline_cost += (row["input_tokens"] or 0) * REFERENCE_INPUT_COST_PER_TOKEN
    baseline_cost += (row["output_tokens"] or 0) * REFERENCE_OUTPUT_COST_PER_TOKEN

actual_cost = df["cost_usd"].sum() + df["escalated_cost_usd"].fillna(0).sum()
savings = baseline_cost - actual_cost
savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric(
    "💰 Cost Savings",
    f"{savings_pct:.1f}%",
    help=f"Compared to a reference baseline of routing every request to {REFERENCE_BASELINE_NAME}"
)
col2.metric("Total Spent (actual)", f"${actual_cost:.4f}")
col3.metric(f"Reference Cost (all-GPT-4o)", f"${baseline_cost:.4f}")

st.caption(
    "⚠️ Reference baseline uses GPT-4o's published per-token pricing applied to actual token counts. "
    "GPT-4o is not called by this system — this is a cost-comparison estimate only."
)
st.markdown("---")

# --- Row 2: Cost over time + Routing distribution ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Cost Over Time")
    df_sorted = df.sort_values("timestamp")
    df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
    fig = px.line(df_sorted, x="timestamp", y="cumulative_cost", title="Cumulative Cost")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🥧 Routing Distribution")
    model_counts = df["routed_model_key"].value_counts().reset_index()
    model_counts.columns = ["model", "count"]
    fig = px.pie(model_counts, names="model", values="count", title="% of Requests per Model")
    st.plotly_chart(fig, use_container_width=True)

# --- Row 3: Quality distribution + Escalation rate ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ Quality Score Distribution")
    quality_df = df[df["quality_score"].notna()]
    if not quality_df.empty:
        fig = px.histogram(quality_df, x="quality_score", nbins=5, title="Quality Scores (1-5)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quality scores logged yet.")

with col2:
    st.subheader("🚨 Escalation Rate Over Time")
    df_sorted["escalated_flag"] = df_sorted["escalated"].astype(int)
    df_sorted["date"] = df_sorted["timestamp"].dt.date
    escalation_by_day = df_sorted.groupby("date")["escalated_flag"].mean().reset_index()
    fig = px.bar(escalation_by_day, x="date", y="escalated_flag", title="Daily Escalation Rate")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# --- Row 4: Cost by model table ---
st.markdown("---")
st.subheader("💵 Cost Breakdown by Model")
by_model_df = pd.DataFrame(stats["by_model"])
st.dataframe(by_model_df, use_container_width=True)

# --- Row 5: Raw request log ---
# --- Row 5: Raw request log ---
st.markdown("---")
st.subheader("📋 Recent Requests")

recent = df.head(50).copy()
recent["prompt_short"] = recent["prompt"].str.slice(0, 80)
recent["answer_short"] = recent["response_text"].fillna("").str.slice(0, 120)

display_cols = ["timestamp", "prompt_short", "answer_short", "predicted_tier",
                "routed_model_key", "cost_usd", "quality_score", "escalated"]
st.dataframe(recent[display_cols], use_container_width=True)

st.subheader("🔍 Read a full answer")
for _, row in recent.head(10).iterrows():
    label = f"{row['timestamp']:%Y-%m-%d %H:%M} | tier {row['predicted_tier']} | {row['routed_model_key']}"
    with st.expander(label):
        st.markdown("**Prompt**")
        st.write(row["prompt"])
        st.markdown("**Answer**")
        st.write(row["response_text"])