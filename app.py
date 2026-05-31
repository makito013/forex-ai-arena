import streamlit as st
import yaml
import sys
import os

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.db_models import init_db, Agent, OpenPosition
from src.engine.financial import FinancialEngine
from src.data.fetcher import MarketDataFetcher
from src.engine.trainer import run_training_session
from src.engine.competition import run_competition
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Forex AI Arena", layout="wide")

st.title("📈 Forex AI Arena - Training Platform")

# Load configuration
@st.cache_resource
def load_engine():
    return FinancialEngine('config.yaml')

@st.cache_resource
def load_data_fetcher():
    engine_config = FinancialEngine('config.yaml').config
    return MarketDataFetcher(engine_config)

@st.cache_resource
def get_db_session():
    return init_db()

engine = load_engine()
fetcher = load_data_fetcher()
session = get_db_session()

st.sidebar.header("Configuration")
st.sidebar.write(f"Leverage: 1:{engine.leverage}")
st.sidebar.write(f"Brokerage Fee: {engine.fee_pct * 100}% on margin")

# Setup Tabs
tab_setup, tab_board, tab_comp, tab_train, tab_data, tab_pos = st.tabs([
    "Arena Setup (Math Test)", "Leaderboard", "Competition Arena ⚔️", "Training Arena 🧠", "Market Data", "Open Positions"
])

with tab_train:
    st.header("🧠 Train New AI Agents")
    st.write("Configure and launch a training session directly from the UI. The agents will play against historical data for the chosen period and asset.")
    
    col_sym, col_per, col_ag = st.columns(3)
    with col_sym:
        pairs = [p['symbol'] for p in engine.config['assets']['pairs']]
        selected_symbol = st.selectbox("Currency Pair / Asset", pairs)
    with col_per:
        period_options = {
            "5 Days (1m chart)": ("5d", "1m"),
            "1 Month (15m chart)": ("1mo", "15m"),
            "3 Months (1h chart)": ("3mo", "1h"),
            "1 Year (1d chart)": ("1y", "1d"),
            "Max Available (1d chart)": ("max", "1d")
        }
        selected_period_label = st.selectbox("Historical Data Period", list(period_options.keys()))
        selected_period, selected_interval = period_options[selected_period_label]
    with col_ag:
        mode = st.radio("Training Mode", ["Create New Agent(s)", "Train Existing Agent"])
        
        existing_agent_name = None
        num_agents = 1
        
        if mode == "Train Existing Agent":
            existing_agents = session.query(Agent).all()
            if existing_agents:
                agent_names = [a.name for a in existing_agents]
                existing_agent_name = st.selectbox("Select Agent", agent_names)
            else:
                st.warning("No existing agents found in database.")
        else:
            num_agents = st.number_input("Number of Agents to Train", min_value=1, max_value=10, value=1)
        
    if st.button("🚀 Start Training Session"):
        if mode == "Train Existing Agent" and not existing_agent_name:
            st.error("Please create an agent first or select 'Create New Agent(s)'.")
        else:
            st.divider()
            overall_status = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            success, results = run_training_session(
                symbol=selected_symbol,
                period=selected_period,
                interval=selected_interval,
                num_agents=num_agents,
                config=engine.config,
                progress_bar=progress_bar,
                status_text=status_text,
                overall_status=overall_status,
                existing_agent_name=existing_agent_name
            )
            
            if success:
                progress_bar.progress(1.0)
                status_text.text("Training Phase Completed.")
                overall_status.success(f"🎉 Successfully trained on {selected_symbol}!")
                st.table(results)
                st.info("Go to the Leaderboard tab to see how they rank globally!")

with tab_comp:
    st.header("⚔️ Competition Arena")
    st.write("Put trained agents against each other on a new dataset! They will NOT learn here, only trade. The score is calculated based on profit % (1pt per 1%) and a penalty for inactivity (-1pt per day without profit).")
    
    existing_agents = session.query(Agent).all()
    if not existing_agents:
        st.warning("No agents available. Train some agents first!")
    else:
        agent_names = [a.name for a in existing_agents]
        selected_competitors = st.multiselect("Select Competitors", agent_names, default=agent_names[:2] if len(agent_names) >= 2 else agent_names)
        
        col_c_sym, col_c_per = st.columns(2)
        with col_c_sym:
            c_pairs = [p['symbol'] for p in engine.config['assets']['pairs']]
            c_symbol = st.selectbox("Competition Asset", c_pairs, key="comp_sym")
        with col_c_per:
            c_period_options = {
                "Recent 5 Days (1m)": ("5d", "1m"),
                "Recent 1 Month (15m)": ("1mo", "15m")
            }
            c_period_label = st.selectbox("Competition Dataset", list(c_period_options.keys()))
            c_period, c_interval = c_period_options[c_period_label]
            
        if st.button("🏆 Start Competition"):
            if not selected_competitors:
                st.error("Select at least one competitor.")
            else:
                st.divider()
                status_text = st.empty()
                progress_bar = st.progress(0.0)
                
                success, comp_results = run_competition(
                    agent_names=selected_competitors,
                    symbol=c_symbol,
                    period=c_period,
                    interval=c_interval,
                    config=engine.config,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                if success:
                    st.success("Competition concluded!")
                    st.table(comp_results)

with tab_data:
    st.subheader("Interactive Market Charts")
    
    col_d_sym, col_d_per = st.columns(2)
    with col_d_sym:
        data_pairs = [p['symbol'] for p in engine.config['assets']['pairs']]
        data_symbol = st.selectbox("Asset", data_pairs, key="data_sym")
    with col_d_per:
        data_period_options = {
            "Recent 5 Days (1m)": ("5d", "1m"),
            "Recent 1 Month (15m)": ("1mo", "15m"),
            "3 Months (1h)": ("3mo", "1h")
        }
        data_period_label = st.selectbox("Historical View", list(data_period_options.keys()), key="data_per")
        d_period, d_interval = data_period_options[data_period_label]
        
    if st.button("Load Chart"):
        with st.spinner("Fetching data and rendering chart..."):
            df = fetcher.fetch_historical_data(data_symbol, period=d_period, interval=d_interval)
            
            if df.empty:
                st.error("No data found or Yahoo Finance restricted this timeframe.")
            else:
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'].squeeze(),
                                high=df['High'].squeeze(),
                                low=df['Low'].squeeze(),
                                close=df['Close'].squeeze())])
                
                fig.update_layout(
                    title=f"{data_symbol} Candlestick Chart ({d_interval})",
                    yaxis_title="Price",
                    xaxis_title="Date/Time",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    st.divider()
    st.subheader("Live Market Prices")
    if st.button("Refresh Quick Prices"):
        with st.spinner("Fetching from yfinance..."):
            prices = fetcher.fetch_current_prices()
            cols = st.columns(len(prices))
            for i, (symbol, price) in enumerate(prices.items()):
                with cols[i]:
                    st.metric(label=symbol, value=f"{price:.5f}" if price else "Error")

with tab_setup:
    st.subheader("Financial Engine Math Test")
    st.info("💡 Note on 'Price': In real trading (MT4/5) you don't pick the opening price, it is executed at the market price automatically. This tab is just a calculator to verify the math logic (Margin, Leverage, and our strict 5% Brokerage Fee rules) at any hypothetical price you type.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_lots = st.number_input("Lots", value=0.01, step=0.01)
    with col2:
        test_price = st.number_input("Hypothetical Open Price", value=1.1000, format="%.4f")
    with col3:
        st.write("Calculations")
        math_result = engine.open_position_math(test_lots, test_price)
        st.json(math_result)

with tab_board:
    st.subheader("AI Leaderboard")
    agents = session.query(Agent).order_by(Agent.score.desc(), Agent.balance.desc()).all()
    if agents:
        st.table([{ "ID": a.id, "Name": a.name, "Global Score 🏆": a.score, "Balance ($)": round(a.balance, 2), "Strategy": a.strategy_type } for a in agents])
    else:
        st.write("No agents in the arena yet.")

with tab_pos:
    st.subheader("Current Open Positions")
    positions = session.query(OpenPosition).all()
    if positions:
        st.table([{
            "Agent": p.agent.name,
            "Symbol": p.symbol,
            "Type": p.position_type,
            "Lots": p.lots,
            "Open Price": p.open_price,
            "Margin": round(p.margin_invested, 2),
            "Fee": round(p.brokerage_fee, 2)
        } for p in positions])
    else:
        st.write("No open positions.")
