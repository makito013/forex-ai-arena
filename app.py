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
import time

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Arena Setup", "Leaderboard", "Open Positions", "Market Data", "Training Arena"])

with tab5:
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

with tab4:
    st.subheader("Live Market Prices")
    if st.button("Refresh Prices"):
        with st.spinner("Fetching from yfinance..."):
            prices = fetcher.fetch_current_prices()
            cols = st.columns(len(prices))
            for i, (symbol, price) in enumerate(prices.items()):
                with cols[i]:
                    st.metric(label=symbol, value=f"{price:.5f}" if price else "Error")

with tab1:
    st.subheader("Financial Engine Math Test")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_lots = st.number_input("Lots", value=0.01, step=0.01)
    with col2:
        test_price = st.number_input("Price", value=1.1000, format="%.4f")
    with col3:
        st.write("Calculations")
        math_result = engine.open_position_math(test_lots, test_price)
        st.json(math_result)

    if st.button("Create Test AI Agent"):
        existing_agent = session.query(Agent).filter_by(name="TestAgent_01").first()
        if not existing_agent:
            new_agent = Agent(name="TestAgent_01", strategy_type="Manual_Test", balance=engine.config['arena']['initial_balance'])
            session.add(new_agent)
            session.commit()
            st.success("Test Agent Created!")
        else:
            st.info("Test Agent already exists.")

with tab2:
    st.subheader("AI Leaderboard")
    agents = session.query(Agent).order_by(Agent.balance.desc()).all()
    if agents:
        st.table([{ "ID": a.id, "Name": a.name, "Strategy": a.strategy_type, "Balance ($)": round(a.balance, 2) } for a in agents])
    else:
        st.write("No agents in the arena yet.")

with tab3:
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
