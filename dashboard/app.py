"""
Three panels powered by the FastAPI endpoints:
  - Pie chart: overall sentiment distribution (/stats)
  - Line chart: sentiment over time (/trends)
  - Live feed: recent scored posts (/recent)

Auto-refreshes every REFRESH_INTERVAL seconds so it updates live
while the pipeline is running.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "10"))

# Colors for consistent branding across charts
COLORS = {
    "NEGATIVE": "#EF4444",   # red
    "NEUTRAL": "#6B7280",    # gray
    "POSITIVE": "#22C55E",   # green
}
LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


# ---- API helpers ----------------------------------------------------------

def fetch_stats() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch /stats: {e}")
        return None


def fetch_trends(bucket: str = "hour", hours: int = 24) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/trends",
            params={"bucket": bucket, "hours": hours},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch /trends: {e}")
        return None


def fetch_recent(limit: int = 50, label: str | None = None) -> dict | None:
    try:
        params = {"limit": limit}
        if label:
            params["label"] = label
        r = requests.get(f"{API_URL}/recent", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch /recent: {e}")
        return None


# ---- page config ----------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Real-Time Sentiment Dashboard")
st.caption("Live Bluesky posts scored by fine-tuned DistilBERT")


# ---- sidebar controls -----------------------------------------------------

with st.sidebar:
    st.header("Controls")

    auto_refresh = st.toggle("Auto-refresh", value=True)
    if auto_refresh:
        st.caption(f"Refreshing every {REFRESH_INTERVAL}s")

    st.divider()

    trend_bucket = st.selectbox(
        "Trend bucket size",
        options=["minute", "hour", "day"],
        index=1,
    )
    trend_hours = st.slider(
        "History (hours)",
        min_value=1,
        max_value=168,
        value=24,
    )

    st.divider()

    feed_limit = st.slider("Posts in feed", min_value=10, max_value=200, value=50)
    feed_filter = st.selectbox(
        "Filter by sentiment",
        options=["All", "POSITIVE", "NEUTRAL", "NEGATIVE"],
        index=0,
    )

    st.divider()
    if st.button("🔄 Refresh now"):
        st.rerun()


# ---- auto-refresh ---------------------------------------------------------

# if auto_refresh:
#     st.empty()
#     import time
#     # st.rerun triggers a full rerun of the script. The fragment below
#     # schedules the next rerun after REFRESH_INTERVAL seconds.
#     # Streamlit's st.auto_refresh is not built-in; we use st.empty + rerun.
#     # This is a well-known pattern for polling dashboards.
#     from streamlit_js_eval import streamlit_js_eval  # noqa: F401
#     # Fallback: use st.rerun with a timer placeholder
#     # (streamlit_js_eval is optional; if not installed, auto-refresh
#     #  works via the manual refresh button instead)


# ---- top metrics row ------------------------------------------------------

stats = fetch_stats()
if stats:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", f"{stats['total']:,}")
    col2.metric("Positive", f"{stats['positive']:,}",
                delta=f"{stats['positive']/max(stats['total'],1)*100:.1f}%")
    col3.metric("Neutral", f"{stats['neutral']:,}",
                delta=f"{stats['neutral']/max(stats['total'],1)*100:.1f}%")
    col4.metric("Negative", f"{stats['negative']:,}",
                delta=f"{stats['negative']/max(stats['total'],1)*100:.1f}%")


# ---- charts row -----------------------------------------------------------

chart_left, chart_right = st.columns([3, 2])

# Pie chart — overall distribution
with chart_right:
    st.subheader("Sentiment Distribution")
    if stats:
        fig_pie = go.Figure(data=[go.Pie(
            labels=LABEL_ORDER,
            values=[stats["negative"], stats["neutral"], stats["positive"]],
            marker=dict(colors=[COLORS[l] for l in LABEL_ORDER]),
            textinfo="label+percent",
            hole=0.4,
        )])
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Line chart — trends over time
with chart_left:
    st.subheader("Sentiment Over Time")
    trends = fetch_trends(bucket=trend_bucket, hours=trend_hours)
    if trends and trends.get("data"):
        fig_line = go.Figure()
        for label in LABEL_ORDER:
            fig_line.add_trace(go.Scatter(
                x=[b["bucket_start"] for b in trends["data"]],
                y=[b[label.lower()] for b in trends["data"]],
                name=label,
                line=dict(color=COLORS[label], width=2),
                mode="lines+markers",
                marker=dict(size=4),
            ))
        fig_line.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            xaxis_title="Time (UTC)",
            yaxis_title=f"Posts per {trend_bucket}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No trend data yet. Run the producer + consumer to generate data.")


# ---- recent posts feed ----------------------------------------------------

st.subheader("Recent Posts")

label_param = None if feed_filter == "All" else feed_filter
recent = fetch_recent(limit=feed_limit, label=label_param)

if recent and recent.get("posts"):
    for post in recent["posts"]:
        label = post["sentiment_label"]
        score = post["sentiment_score"]
        body = post["body"] or post["title"] or "(no text)"
        ts = post["created_utc"]

        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except (ValueError, AttributeError):
            time_str = str(ts)

        # Color-coded badge
        color = COLORS.get(label, "#6B7280")
        badge = f":{color[1:]}[**{label}**] ({score:.0%})"

        with st.container():
            col_text, col_meta = st.columns([4, 1])
            with col_text:
                st.markdown(f"{body}")
            with col_meta:
                st.markdown(f"**{label}** {score:.0%}")
                st.caption(time_str)
            st.divider()
else:
    st.info("No posts yet. Run the producer + consumer to generate data.")


# ---- auto-refresh via streamlit fragment ----------------------------------

if auto_refresh:
    import time as _time
    _time.sleep(REFRESH_INTERVAL)
    st.rerun()
