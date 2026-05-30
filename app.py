import streamlit as st
import yaml
import sys
import os

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.db_models import init_db, Agent, OpenPosition
from src.engine.financial import FinancialEngine
from src.data.fetcher import MarketDataFetcher
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
tab1, tab2, tab3, tab4 = st.tabs(["Arena Setup (Test Engine)", "Leaderboard", "Open Positions", "Market Data"])

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
