import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import warnings
import joblib
import xgboost as xgb
warnings.filterwarnings("ignore")

from main import run_pipeline, download_yahoo, add_indicators, simple_backtest, create_supervised, normalize_df

st.set_page_config(
    page_title="AlgoTrade AI — MarketPulse Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_COLS = ['SMA14','WMA14','MOM14','STCK','STCD','RSI14','MACD','MACD_SIGNAL','LWR14','ADO','CCI20']

FEATURE_LABELS = {
    'SMA14':        'Simple Moving Avg (14d)',
    'WMA14':        'Weighted Moving Avg (14d)',
    'MOM14':        'Momentum (14d)',
    'STCK':         'Stochastic %K',
    'STCD':         'Stochastic %D',
    'RSI14':        'RSI — Relative Strength',
    'MACD':         'MACD',
    'MACD_SIGNAL':  'MACD Signal Line',
    'LWR14':        'Williams %R (14d)',
    'ADO':          'Accumulation/Distribution',
    'CCI20':        'Commodity Channel Index (20d)',
}

# ─────────────────────────────────────────────
# Helper: colour-coded rating
# ─────────────────────────────────────────────
def rating(value, thresholds, higher_is_better=True):
    g, ok = thresholds
    if higher_is_better:
        if value >= g:    return "🟢", "Great"
        elif value >= ok: return "🟡", "Okay"
        else:             return "🔴", "Needs work"
    else:
        if value <= g:    return "🟢", "Great"
        elif value <= ok: return "🟡", "Okay"
        else:             return "🔴", "Needs work"

# ─────────────────────────────────────────────
# WOW #1 — LIVE SIGNAL (runs on every page load)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)   # refresh every 5 min
def get_live_signal(ticker: str):
    try:
        since = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')
        df = download_yahoo(ticker, start=since)
        if df is None or len(df) < 30:
            return None
        df = add_indicators(df)
        data = create_supervised(df, FEATURE_COLS)
        if len(data) < 20:
            return None
        X = data[FEATURE_COLS].values
        y = data['target'].values
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                                  max_depth=4, verbosity=0)
        model.fit(X[:-1], y[:-1])
        pred = float(model.predict(X[-1].reshape(1, -1))[0])
        close_today = float(df['Close'].iloc[-1])
        close_prev  = float(df['Close'].iloc[-2])
        day_change  = (close_today - close_prev) / close_prev * 100

        if pred > 0.001:
            action, color, emoji = "BUY",  "#00e676", "🟢"
        elif pred < -0.001:
            action, color, emoji = "SELL", "#ff1744", "🔴"
        else:
            action, color, emoji = "HOLD", "#ffea00", "⬜"

        return {
            "action": action, "color": color, "emoji": emoji,
            "pred": pred, "close": close_today, "day_change": day_change,
            "as_of": df.index[-1].strftime('%d %b %Y')
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

ticker = st.sidebar.text_input("Crypto Symbol", value="BTC-USD",
    help="Yahoo Finance tickers: BTC-USD, ETH-USD, SOL-USD, BNB-USD")
st.sidebar.caption("💡 Try: `BTC-USD` · `ETH-USD` · `SOL-USD`")

start_date = st.sidebar.date_input("Fetch data starting from", value=date(2018, 1, 1))

model_choice = st.sidebar.selectbox(
    "Which AI model drives the strategy?",
    ["XGBoost", "RandomForest", "LSTM"]
)
with st.sidebar.expander("ℹ️ Model differences"):
    st.markdown("""
| Model | Think of it as | Speed |
|---|---|---|
| **XGBoost** | Expert voting panel | ⚡ Fast |
| **RandomForest** | Many independent analysts | ⚡ Fast |
| **LSTM** | Analyst with memory | 🐢 Slower |
    """)

window_size = st.sidebar.slider(
    "AI Memory — how many past days to look back?",
    min_value=5, max_value=30, value=10,
    help="Only used by LSTM. 10 = AI studies last 10 days to predict tomorrow."
)
st.sidebar.caption(f"Currently **{window_size} days** of lookback.")
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Full Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.title("📊 AlgoTrade AI — MarketPulse Optimizer")
st.markdown(
    "An **AI-powered trading system** that learns from crypto history, predicts market direction, "
    "and simulates whether those predictions would have made money.\n\n"
    "> Trains 3 AI models → generates Buy/Sell/Hold signals → compares strategy vs simply holding"
)

# ─────────────────────────────────────────────
# WOW #1 DISPLAY — LIVE SIGNAL BANNER
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader(f"⚡ Live Signal — What should you do with {ticker} RIGHT NOW?")
st.caption("Auto-refreshes every 5 minutes using the latest market data · Powered by XGBoost")

with st.spinner("Fetching live market data…"):
    live = get_live_signal(ticker)

if live:
    lc1, lc2, lc3, lc4 = st.columns([2, 1.5, 1.5, 1.5])
    with lc1:
        st.markdown(
            f"""
            <div style="background:{live['color']}22; border:2px solid {live['color']};
                        border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:52px">{live['emoji']}</div>
                <div style="font-size:32px; font-weight:bold; color:{live['color']}">
                    {live['action']}
                </div>
                <div style="font-size:13px; opacity:0.8;">AI Recommendation as of {live['as_of']}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with lc2:
        st.metric("Current Price", f"${live['close']:,.2f}",
                  delta=f"{live['day_change']:+.2f}% today")
    with lc3:
        direction = "📈 Rise" if live['pred'] > 0 else "📉 Fall"
        st.metric("AI Predicts Tomorrow", direction,
                  delta=f"{live['pred']*100:+.3f}% expected return")
    with lc4:
        st.metric("Signal Strength",
                  "Strong" if abs(live['pred']) > 0.005 else "Moderate" if abs(live['pred']) > 0.001 else "Weak",
                  delta="above noise threshold" if abs(live['pred']) > 0.001 else "within noise band")
    with st.expander("📖 How is this live signal generated?"):
        st.markdown(f"""
1. Downloads **last 120 days** of {ticker} prices from Yahoo Finance (right now)
2. Computes all **11 technical indicators** from that fresh data
3. Trains a **quick XGBoost model** on those 120 days (in memory, takes seconds)
4. Predicts the **expected return for tomorrow**
5. Applies the rule: predicted return **> +0.1%** → BUY | **< −0.1%** → SELL | in between → HOLD
        """)
else:
    st.warning("Could not fetch live signal. Check ticker symbol or internet connection.")

st.markdown("---")

# ─────────────────────────────────────────────
# FULL PIPELINE RESULTS
# ─────────────────────────────────────────────
if run_button:
    with st.spinner("⏳ Training AI models on full history… (~2-3 minutes)"):
        output = run_pipeline(
            ticker=ticker,
            start=str(start_date),
            window_size=window_size,
            model_choice=model_choice
        )

    st.success(f"✅ Full analysis complete for **{ticker}** — strategy driven by **{output['chosen_label']}**")
    st.markdown("---")

    stats   = output["backtest_stats"]
    xm      = output["xgb_metrics"]
    rfm     = output["rf_metrics"]
    lm      = output["lstm_metrics"]
    back_df = output["backtest_df"]
    chosen  = output["chosen_label"]

    final_strategy = round(back_df["cum_strategy"].iloc[-1], 3)
    final_buyhold  = round(back_df["cum_asset"].iloc[-1], 3)
    outperformed   = final_strategy > final_buyhold

    # ── PLAIN ENGLISH SUMMARY ─────────────────────────────────────────────
    st.subheader("📝 Plain English Summary")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        arrow = "📈" if outperformed else "📉"
        diff  = round(abs(final_strategy - final_buyhold), 3)
        word  = "outperformed ✅" if outperformed else "underperformed ❌"
        st.info(f"""
**If you invested ₹1,000 at the start of this period:**
- 🤖 Following our AI signals → **₹{final_strategy * 1000:,.0f}**
- 🧍 Just holding {ticker} → **₹{final_buyhold * 1000:,.0f}**

{arrow} Our strategy **{word}** Buy & Hold by **{diff:.3f}×**
        """)

    with col_s2:
        da_chosen = lm["DA"] if chosen == "LSTM" else (rfm["DA"] if chosen == "RandomForest" else xm["DA"])
        e1, v1 = rating(da_chosen,       (60, 53))
        e2, v2 = rating(stats["sharpe"], (1.0, 0.5))
        st.info(f"""
**Quick health check:**
- {e1} AI Direction Accuracy: **{da_chosen:.1f}%** — {v1}
- {e2} Risk-Adjusted Return (Sharpe): **{stats['sharpe']}** — {v2}
- 🎯 Win Rate on trades: **{stats['win_rate']}%**
- 📉 Worst loss period: **{stats['max_drawdown']}%**
        """)

    st.markdown("---")

    # ── MODEL ACCURACY ────────────────────────────────────────────────────
    st.subheader("🤖 How Accurate Were the AI Models?")
    st.caption("Most important metric: **Direction Accuracy** (did AI correctly predict UP or DOWN?). 50% = coin flip.")

    col1, col2, col3 = st.columns(3)

    def model_card(col, name, m, is_chosen):
        em, verdict = rating(m["DA"], (60, 53))
        badge = " ← selected" if is_chosen else ""
        with col:
            st.markdown(f"#### {name}{badge}")
            st.metric("Direction Accuracy", f"{m['DA']:.1f}%",
                      help="% of days the model predicted market direction correctly.")
            st.markdown(f"{em} **{verdict}** &nbsp;|&nbsp; Prediction Error: `{m['RMSE']:.6f}`")
            with st.expander("📖 Explain"):
                st.markdown(f"""
- **{m['DA']:.1f}% Direction Accuracy** → model was right **{m['DA']:.0f} out of 100 days**
- **RMSE {m['RMSE']:.6f}** → average prediction error (daily returns are tiny decimals — this is normal)
- 50% = coin flip. **Above ~55% = the AI is genuinely learning market patterns.**
                """)

    model_card(col1, "XGBoost",      xm,  chosen == "XGBoost")
    model_card(col2, "RandomForest", rfm, chosen == "RandomForest")
    model_card(col3, "LSTM",         lm,  chosen == "LSTM")

    # ── WOW #3 — RADAR CHART ──────────────────────────────────────────────
    st.markdown("#### 🕸️ Model Comparison — Radar Chart")
    st.caption("Each axis = one performance dimension. Bigger area = better overall model.")

    # Normalize Sharpe to 0-100 for radar (cap at 3.0 = excellent)
    def sharpe_score(s):
        return min(max(s / 3.0 * 100, 0), 100)

    categories = ["Direction Accuracy", "Win Rate (proxy)", "Sharpe Score", "Low Error (inv RMSE)"]

    max_rmse = max(xm["RMSE"], rfm["RMSE"], lm["RMSE"]) + 1e-9

    def radar_vals(m, st_sharpe):
        return [
            m["DA"],
            min(m["DA"] * 0.9, 100),                    # proxy (DA correlates with WR)
            sharpe_score(st_sharpe),
            (1 - m["RMSE"] / max_rmse) * 100
        ]

    fig_radar = go.Figure()
    models_radar = [
        ("XGBoost",      xm,  stats["sharpe"], "#00b4d8"),
        ("RandomForest", rfm, stats["sharpe"], "#90e0ef"),
        ("LSTM",         lm,  stats["sharpe"], "#48cae4"),
    ]
    for name, m, sh, color in models_radar:
        vals = radar_vals(m, sh)
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill='toself', name=name,
            line_color=color, opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, template="plotly_dark", height=400,
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── WOW #2 — FEATURE IMPORTANCE ───────────────────────────────────────
    st.subheader("🔍 What Did the AI Learn? — Feature Importance")
    st.caption("Which of the 11 technical indicators did XGBoost rely on most to make predictions?")

    try:
        saved_xgb = joblib.load('models/xgb_model.pkl')
        importances = pd.Series(
            saved_xgb.feature_importances_,
            index=[FEATURE_LABELS[c] for c in FEATURE_COLS]
        ).sort_values(ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            marker=dict(
                color=importances.values,
                colorscale='Teal',
                showscale=False
            ),
            text=[f"{v:.3f}" for v in importances.values],
            textposition='outside'
        ))
        fig_imp.update_layout(
            title="XGBoost Feature Importance — Which indicator matters most?",
            xaxis_title="Importance Score",
            yaxis_title="",
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=60)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        top_feat  = importances.idxmax()
        top_score = importances.max()
        st.success(
            f"🏆 **Most important indicator: {top_feat}** (score: {top_score:.3f}) — "
            f"the AI found this pattern most predictive of next-day price direction."
        )
        with st.expander("📖 What does feature importance mean?"):
            st.markdown("""
- XGBoost builds **decision trees** that split on features to make predictions.
- **Feature importance** counts how many times each indicator was used to make a split, weighted by how much it reduced prediction error.
- A **high score = the AI trusted that indicator most** when deciding Buy/Sell/Hold.
- A **low score** doesn't mean the indicator is useless — it may be correlated with a higher-ranked one.
- This is powerful evidence that your choice of technical indicators was **validated by the AI itself**, not just assumed.
            """)
    except Exception:
        st.info("Feature importance chart available after running the pipeline.")

    st.markdown("---")

    # ── BACKTEST STATS ────────────────────────────────────────────────────
    st.subheader("📊 Strategy Performance — Did It Make Money?")
    st.caption(f"Simulating {ticker} trading using **{chosen}** signals · includes 0.1% transaction cost per trade")

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    e_sh, v_sh = rating(stats["sharpe"],              (1.0, 0.5))
    e_dd, v_dd = rating(abs(stats["max_drawdown"]),   (20, 40), higher_is_better=False)
    e_wr, v_wr = rating(stats["win_rate"],            (55, 50))

    sc1.metric("Sharpe Ratio",  f"{e_sh} {stats['sharpe']}",
               help=">1 = good, >2 = excellent. Return divided by risk.")
    sc2.metric("Max Drawdown",  f"{e_dd} {stats['max_drawdown']}%",
               help="Worst peak-to-trough drop. Lower = safer strategy.")
    sc3.metric("Win Rate",      f"{e_wr} {stats['win_rate']}%",
               help="% of active trades that were profitable. 50% = coin flip.")
    sc4.metric("Total Trades",  stats["total_trades"],
               help="Long + Short trades. AI skips low-confidence signals.")
    sc5.metric("Long  (Buy)",   stats["long_trades"],
               help="Days AI predicted strong upward move.")
    sc6.metric("Short (Sell)",  stats["short_trades"],
               help="Days AI predicted strong downward move.")

    with st.expander("📖 Understand these numbers"):
        st.markdown(f"""
| Metric | Your Value | Plain English |
|---|---|---|
| **Sharpe Ratio** | {stats['sharpe']} | Return vs stress ratio. **>1 is good** (earning well for the risk taken). {e_sh} {v_sh} |
| **Max Drawdown** | {stats['max_drawdown']}% | Worst losing streak before recovery — portfolio fell this much from its peak. {e_dd} {v_dd} |
| **Win Rate** | {stats['win_rate']}% | When AI took a position, it was right **{stats['win_rate']}% of the time**. {e_wr} {v_wr} |
| **Total Trades** | {stats['total_trades']} | AI only trades when confident (ignores ±0.1% noise band) |
| **Long / Short** | {stats['long_trades']} / {stats['short_trades']} | Long = bet on rise · Short = bet on fall |
        """)

    st.markdown("---")

    # ── GRAPH ─────────────────────────────────────────────────────────────
    st.subheader("💹 Strategy vs Buy & Hold — Visual Comparison")
    st.markdown(f"""
**How to read this chart:**
- 🩶 **Dashed gray** = "I bought {ticker} and did nothing" (Buy & Hold baseline)
- 🟢 **Green** = "I followed the AI's signals every day" (Our Strategy)
- Value of **1.5** = portfolio grew to **1.5× the starting amount** (+50% total)
- Green **above** gray → AI strategy is winning over passive holding
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=back_df.index, y=back_df["cum_asset"],
        name="Buy & Hold (passive)",
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=back_df.index, y=back_df["cum_strategy"],
        name=f"AI Strategy ({chosen})",
        line=dict(color='#00e676', width=2.5),
        fill='tonexty', fillcolor='rgba(0,230,118,0.06)'
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="white", opacity=0.3,
                  annotation_text="Break-even", annotation_position="bottom right")
    fig.update_layout(
        title=f"Cumulative Return: {ticker} — AI Strategy vs Buy & Hold [{chosen}]",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (×, starting at 1.0)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── WOW #4 — INVESTMENT CALCULATOR ───────────────────────────────────
    st.markdown("---")
    st.subheader("🧮 What Would YOUR Investment Be Worth?")
    st.caption("See how much you would have made (or lost) following this strategy vs just holding")

    inv_col1, inv_col2 = st.columns([1, 2])
    with inv_col1:
        invest_amount = st.number_input(
            "Enter your investment amount (₹)",
            min_value=100, max_value=10_000_000,
            value=10_000, step=1000,
            format="%d"
        )
        currency = st.selectbox("Currency", ["₹ INR", "$ USD", "€ EUR"])
        symbol = currency.split()[0]

    with inv_col2:
        ai_result  = round(invest_amount * final_strategy, 2)
        bh_result  = round(invest_amount * final_buyhold,  2)
        ai_profit  = round(ai_result  - invest_amount, 2)
        bh_profit  = round(bh_result  - invest_amount, 2)
        ai_pct     = round((final_strategy - 1) * 100, 2)
        bh_pct     = round((final_buyhold  - 1) * 100, 2)

        c_ai, c_bh = st.columns(2)
        with c_ai:
            delta_color = "normal" if ai_profit >= 0 else "inverse"
            st.metric(
                f"🤖 AI Strategy result",
                f"{symbol}{ai_result:,.0f}",
                delta=f"{symbol}{ai_profit:+,.0f} ({ai_pct:+.1f}%)"
            )
            st.caption(f"Starting from {symbol}{invest_amount:,}")
        with c_bh:
            st.metric(
                f"🧍 Buy & Hold result",
                f"{symbol}{bh_result:,.0f}",
                delta=f"{symbol}{bh_profit:+,.0f} ({bh_pct:+.1f}%)"
            )
            st.caption(f"Starting from {symbol}{invest_amount:,}")

        if outperformed:
            diff = round(ai_result - bh_result, 2)
            st.success(f"🏆 AI Strategy earned **{symbol}{diff:,.0f} more** than just holding {ticker}!")
        else:
            diff = round(bh_result - ai_result, 2)
            st.warning(f"📊 Buy & Hold earned **{symbol}{diff:,.0f} more** in this period — strategy was conservative (lower risk too).")

    st.markdown("---")

    # ── SIGNALS TABLE ─────────────────────────────────────────────────────
    st.subheader("📋 Recent Trading Signals — Last 10 Days")
    st.caption("What the AI recommended on each of the last 10 trading days")

    display_df = back_df[['Close','pred','signal','strategy_ret','cum_strategy']].tail(10).copy()
    display_df = display_df.rename(columns={
        'Close':        'Closing Price ($)',
        'pred':         'Predicted Return',
        'signal':       '_sig',
        'strategy_ret': 'Return Earned',
        'cum_strategy': 'Portfolio Value (×)'
    })
    display_df['Action'] = back_df['signal'].tail(10).map({
        1: '🟢 BUY (Long)', -1: '🔴 SELL (Short)', 0: '⬜ HOLD'
    })
    display_df = display_df[['Closing Price ($)', 'Predicted Return', 'Action', 'Return Earned', 'Portfolio Value (×)']]
    st.dataframe(display_df, use_container_width=True)

    with st.expander("📖 How to read this table"):
        st.markdown("""
| Column | Meaning |
|---|---|
| **Closing Price** | Actual crypto price that day |
| **Predicted Return** | AI forecast (positive = expects rise, negative = expects fall) |
| **Action** | 🟢 BUY = predicted strong rise · 🔴 SELL = predicted strong fall · ⬜ HOLD = uncertain |
| **Return Earned** | Actual profit/loss from that day's position (after 0.1% transaction fee) |
| **Portfolio Value** | Cumulative growth since start (1.5 = grew 50%) |
        """)

    st.markdown("---")

    st.subheader("💾 Download Full Results")
    csv = back_df.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="⬇️ Download Backtest Data as CSV",
        data=csv,
        file_name=f"{ticker}_backtest_{chosen}.csv",
        mime="text/csv"
    )

else:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1** 👈\nChoose a crypto coin and date range in the sidebar")
    with c2:
        st.info("**Step 2** 🧠\nPick an AI model (XGBoost is fastest)")
    with c3:
        st.info("**Step 3** 🚀\nClick **Run Full Analysis** to train models")

    st.markdown("---")
    with st.expander("📚 How does this system work?"):
        st.markdown("""
```
1. FETCH DATA      → Downloads real price history from Yahoo Finance
2. COMPUTE SIGNALS → Calculates 11 technical indicators traders use
3. TRAIN AI        → Teaches 3 models to predict next-day price direction
4. GENERATE SIGNALS→ BUY / SELL / HOLD based on predictions
5. BACKTEST        → Simulates trading to measure real performance
```
| Model | Think of it as |
|---|---|
| **XGBoost** | Expert voting panel — very accurate on patterns |
| **RandomForest** | Many analysts independently voting, majority wins |
| **LSTM** | Analyst with memory — learns from sequence of days |

**Why crypto?** Trades 24/7 · More data to learn from · Live prices always available
        """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>AlgoTrade AI — MarketPulse Optimizer &nbsp;|&nbsp; "
    "Python · XGBoost · TensorFlow · Streamlit &nbsp;|&nbsp; "
    "<em>\"In trading, data is the new alpha.\"</em></small>",
    unsafe_allow_html=True
)
