import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Import from your main script
from main import run_pipeline, download_yahoo, add_indicators, simple_backtest

# Streamlit page config
st.set_page_config(
    page_title="AlgoTrade AI — MarketPulse Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### Select Parameters")

ticker = st.sidebar.text_input("Stock Symbol (Yahoo Finance)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date(2015, 1, 1))
window_size = st.sidebar.slider("LSTM Window Size", 5, 30, 10)
model_choice = st.sidebar.selectbox("Model for Prediction", ["XGBoost", "RandomForest", "LSTM"])
run_button = st.sidebar.button("🚀 Run Pipeline")

st.sidebar.markdown("---")
st.sidebar.info(
    "📘 **Tip:** Use a Yahoo Finance ticker like `AAPL`, `MSFT`, or `GOOG`.\n"
    "Model will train & backtest automatically."
)

# -----------------------------
# Main Page Header
# -----------------------------
st.title("📊 AlgoTrade AI — MarketPulse Optimizer")
st.markdown("""
A **FinTech AI System** that uses Machine Learning to analyze stock market trends, 
compute technical indicators, and simulate trading strategies.

Built with 🧠 XGBoost, 🌲 RandomForest, and 🔁 LSTM Deep Learning.
""")

# -----------------------------
# Run Section
# -----------------------------
if run_button:
    with st.spinner("⏳ Running end-to-end pipeline... please wait (this may take a few minutes)"):
        output = run_pipeline(ticker=ticker, start=str(start_date), window_size=window_size)

    # Display model metrics
    st.success(f"✅ Completed pipeline for `{ticker}`")

    st.subheader("📈 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("XGBoost RMSE", round(output["xgb_metrics"]["RMSE"], 6))
        st.metric("XGBoost MAPE", f"{output['xgb_metrics']['MAPE']:.3f}%")

    with col2:
        st.metric("RandomForest RMSE", round(output["rf_metrics"]["RMSE"], 6))
        st.metric("RandomForest MAPE", f"{output['rf_metrics']['MAPE']:.3f}%")

    with col3:
        st.metric("LSTM RMSE", round(output["lstm_metrics"]["RMSE"], 6))
        st.metric("LSTM MAPE", f"{output['lstm_metrics']['MAPE']:.3f}%")

    # -----------------------------
    # Cumulative Returns Plot
    # -----------------------------
    st.subheader("💹 Strategy Backtest Visualization")

    back_df = output["backtest_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=back_df.index, y=back_df["cum_asset"], name="Buy & Hold", line=dict(color='gray', width=2)))
    fig.add_trace(go.Scatter(x=back_df.index, y=back_df["cum_strategy"], name="Strategy (XGBoost)", line=dict(color='lime', width=2)))

    fig.update_layout(
        title=f"Cumulative Returns Comparison — {ticker}",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Show prediction table
    # -----------------------------
    st.subheader("📋 Recent Predictions & Signals")
    st.dataframe(
        back_df[['Close', 'pred', 'signal', 'strategy_ret', 'cum_strategy']].tail(10),
        use_container_width=True
    )

    # -----------------------------
    # Download section
    # -----------------------------
    st.subheader("💾 Download Results")
    csv = back_df.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="⬇️ Download Backtest CSV",
        data=csv,
        file_name=f"{ticker}_backtest.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Pipeline** to begin analysis.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
### 🧠 About This App
**AlgoTrade AI** — MarketPulse Optimizer integrates traditional technical analysis with modern AI models to 
generate data-driven trading insights.  
Developed using **Python**, **scikit-learn**, **XGBoost**, **TensorFlow**, and **Streamlit**.

> “In trading, data is the new alpha.”  
> — MarketPulse Research Team
""")
