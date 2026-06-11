import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bitcoin Sentiment vs Trader Performance",
    layout="wide"
)

st.title("📈 Bitcoin Market Sentiment vs Trader Performance")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    fear = pd.read_csv("fear_greed_index.csv")
    trades = pd.read_csv("historical_data.csv")
    return fear, trades

try:
    fear, trades = load_data()

    st.success("Datasets loaded successfully!")

    # Debug Section
    with st.expander("Dataset Information"):
        st.write("Fear & Greed Columns:")
        st.write(fear.columns.tolist())

        st.write("Trader Data Columns:")
        st.write(trades.columns.tolist())

        st.write("Fear Shape:", fear.shape)
        st.write("Trades Shape:", trades.shape)

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ----------------------------
# Date Processing
# ----------------------------

try:
    fear["date"] = pd.to_datetime(fear["date"], errors="coerce")

    trades["Timestamp IST"] = pd.to_datetime(
        trades["Timestamp IST"],
        errors="coerce"
    )

    trades["date"] = trades["Timestamp IST"].dt.normalize()

except Exception as e:
    st.error(f"Date conversion error: {e}")
    st.stop()

# ----------------------------
# Merge
# ----------------------------

try:
    # Try exact date merge first
    merged = trades.merge(
        fear[["date", "classification"]],
        on="date",
        how="inner"
    )

    # If no direct matches, align each trade to the most recent prior fear date
    if len(merged) == 0:
        fear_sorted = (
            fear[["date", "classification"]]
            .dropna(subset=["date"])
            .sort_values("date")
        )
        trades_sorted = trades.dropna(subset=["date"]).sort_values("date")

        merged = pd.merge_asof(
            trades_sorted,
            fear_sorted,
            on="date",
            direction="backward"
        )

        st.info("Performed asof merge (matched each trade to nearest prior fear date).")

    st.write("### Merge Information")
    st.write("Merged Shape:", merged.shape)

except Exception as e:
    st.error(f"Merge error: {e}")
    st.stop()

if len(merged) == 0:
    st.warning("No matching dates found between datasets.")
    st.stop()

# ----------------------------
# Metrics
# ----------------------------

st.write("## Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Fear Records", len(fear))

with col2:
    st.metric("Trade Records", len(trades))

with col3:
    st.metric("Merged Records", len(merged))

# ----------------------------
# PnL Analysis
# ----------------------------

if "Closed PnL" not in merged.columns:
    st.error(
        "Column 'Closed PnL' not found. Check column names."
    )
    st.stop()

st.write("## Average PnL by Sentiment")

avg_pnl = (
    merged.groupby("classification")["Closed PnL"]
    .mean()
    .sort_values()
)

fig1, ax1 = plt.subplots(figsize=(8, 4))
avg_pnl.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Average PnL")
st.pyplot(fig1)

# ----------------------------
# Win Rate
# ----------------------------

merged["win"] = merged["Closed PnL"] > 0

win_rate = (
    merged.groupby("classification")["win"]
    .mean()
    * 100
)

st.write("## Win Rate by Sentiment")

fig2, ax2 = plt.subplots(figsize=(8, 4))
win_rate.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Win Rate (%)")
st.pyplot(fig2)

# ----------------------------
# Sentiment Summary
# ----------------------------

st.write("## Sentiment Summary")

summary = merged.groupby("classification").agg(
    Trades=("Closed PnL", "count"),
    Avg_PnL=("Closed PnL", "mean"),
    Total_PnL=("Closed PnL", "sum"),
    Win_Rate=("win", "mean")
)

summary["Win_Rate"] *= 100

st.dataframe(summary)

# ----------------------------
# Top Traders
# ----------------------------

if "Account" in merged.columns:

    st.write("## Top 10 Traders")

    top_traders = (
        merged.groupby("Account")["Closed PnL"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_traders)

# ----------------------------
# Sample Data
# ----------------------------

st.write("## Sample Merged Data")

st.dataframe(merged.head(100))