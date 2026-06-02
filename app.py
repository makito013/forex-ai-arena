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
    
    t_data_source = st.radio("Data Source", ["Yahoo Finance (Online)", "Local CSV (Offline)"], key="train_ds")
    
    selected_symbol = "CSV"
    selected_period = "max"
    selected_interval = "15m"
    use_csv = False
    csv_paths = []
    sentiment_csv_paths = []
    
    col_sym, col_per, col_ag = st.columns(3)
    
    if t_data_source == "Yahoo Finance (Online)":
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
    else:
        use_csv = True
        with col_sym:
            csv_folder = "data/historical"
            if not os.path.exists(csv_folder):
                os.makedirs(csv_folder)
            csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            if csv_files:
                selected_csvs = st.multiselect("Select Price CSV(s)", csv_files, default=[csv_files[0]] if csv_files else [])
                csv_paths = [os.path.join(csv_folder, f) for f in selected_csvs]
            else:
                st.warning("No CSV files found in 'data/historical'.")
        with col_per:
            csv_interval_options = {
                "1 Minute (1m)": "1m",
                "5 Minutes (5m)": "5m",
                "15 Minutes (15m)": "15m",
                "30 Minutes (30m)": "30m",
                "1 Hour (1h)": "1h",
                "1 Day (1d)": "1d"
            }
            csv_int_label = st.selectbox("CSV Timeframe", list(csv_interval_options.keys()))
            selected_interval = csv_interval_options[csv_int_label]

    # Optional Sentiment Data
    with st.expander("Optional: Include News Sentiment"):
        st.write("Select one or more CSVs with 'Datetime' and 'Sentiment' (score -1 to 1).")
        news_folder = "data/news"
        if not os.path.exists(news_folder): os.makedirs(news_folder)
        news_files = [f for f in os.listdir(news_folder) if f.endswith('.csv')]
        if news_files:
            selected_news = st.multiselect("Select Sentiment CSV(s)", news_files)
            sentiment_csv_paths = [os.path.join(news_folder, f) for f in selected_news]
        else:
            st.info("No news CSV files found in 'data/news'.")
            
    with col_ag:
        mode = st.radio("Training Mode", ["Create New Agent(s)", "Retrain Existing Agent(s)", "Deep Evolutionary Training 🧬"])
        
        existing_agent_names = None
        num_agents = 1
        target_epochs = 10
        
        if mode == "Retrain Existing Agent(s)":
            existing_agents = session.query(Agent).all()
            if existing_agents:
                agent_names = [a.name for a in existing_agents]
                existing_agent_names = st.multiselect("Select Agent(s) to Retrain", agent_names, default=None)
            else:
                st.warning("No existing agents found in database.")
        elif mode == "Deep Evolutionary Training 🧬":
            st.info("Survival of the fittest: Many agents will be trained, but only those that end with profit will be saved.")
            num_agents = st.number_input("Population Size (Candidates)", min_value=1, max_value=50, value=5)
            target_epochs = st.slider("Target Epochs (Passes over data)", min_value=1, max_value=100, value=20)
        else:
            num_agents = st.number_input("Number of New Agents to Train", min_value=1, max_value=10, value=1)
        
    if st.button("🚀 Start Training Session"):
        if mode == "Retrain Existing Agent(s)" and not existing_agent_names:
            st.error("Please select at least one agent to retrain.")
        elif use_csv and not csv_paths:
            st.error("Please place a CSV file in 'data/historical' and select it.")
        else:
            st.divider()
            overall_status = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Live Monitor Container
            st.write("---")
            st.subheader("📊 Live Training Monitor")
            mon_col1, mon_col2 = st.columns(2)
            trade_metric = mon_col1.empty()
            pnl_metric = mon_col2.empty()
            
            if mode == "Deep Evolutionary Training 🧬":
                from src.engine.trainer import run_deep_evolutionary_training
                success, results = run_deep_evolutionary_training(
                    symbol=selected_symbol,
                    period=selected_period,
                    interval=selected_interval,
                    num_agents=num_agents,
                    config=engine.config,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    overall_status=overall_status,
                    trade_metric=trade_metric,
                    pnl_metric=pnl_metric,
                    use_csv=use_csv,
                    csv_paths=csv_paths,
                    sentiment_csv_paths=sentiment_csv_paths,
                    target_epochs=target_epochs
                )
            else:
                success, results = run_training_session(
                    symbol=selected_symbol,
                    period=selected_period,
                    interval=selected_interval,
                    num_agents=num_agents,
                    config=engine.config,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    overall_status=overall_status,
                    trade_metric=trade_metric,
                    pnl_metric=pnl_metric,
                    existing_agent_names=existing_agent_names,
                    use_csv=use_csv,
                    csv_paths=csv_paths,
                    sentiment_csv_paths=sentiment_csv_paths
                )
            
            if success:
                progress_bar.progress(1.0)
                status_text.text("Training Phase Completed.")
                overall_status.success(f"🎉 Successfully trained!")
                st.table(results)
                st.info("Go to the Leaderboard tab to see how they rank globally!")

with tab_comp:
    st.header("⚔️ Competition Arena")
    st.write("Put trained agents against each other on a new dataset! They will NOT learn here, only trade.")
    
    existing_agents = session.query(Agent).all()
    if not existing_agents:
        st.warning("No agents available. Train some agents first!")
    else:
        agent_names = [a.name for a in existing_agents]
        
        col_c_sel, col_c_ds = st.columns([1, 2])
        with col_c_sel:
            if st.checkbox("Select All Agents", value=True):
                selected_competitors = st.multiselect("Select Competitors", agent_names, default=agent_names)
            else:
                selected_competitors = st.multiselect("Select Competitors", agent_names)
        
        with col_c_ds:
            c_data_source = st.radio("Competition Data Source", ["Yahoo Finance (Online)", "Local CSV (Offline)"], key="comp_ds", horizontal=True)
        
        c_symbol = "CSV"
        c_period = "max"
        c_interval = "15m"
        c_use_csv = False
        c_csv_path = None
        
        col_c_sym, col_c_per = st.columns(2)
        
        if c_data_source == "Yahoo Finance (Online)":
            with col_c_sym:
                c_pairs = [p['symbol'] for p in engine.config['assets']['pairs']]
                c_symbol = st.selectbox("Competition Asset", c_pairs, key="comp_sym")
            with col_c_per:
                c_period_options = {
                    "Recent 5 Days (1m)": ("5d", "1m"),
                    "Recent 1 Month (15m)": ("1mo", "15m"),
                    "Recent 6 Months (1h)": ("6mo", "1h")
                }
                c_period_label = st.selectbox("Competition Dataset", list(c_period_options.keys()))
                c_period, c_interval = c_period_options[c_period_label]
        else:
            c_use_csv = True
            with col_c_sym:
                csv_folder = "data/historical"
                csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')] if os.path.exists(csv_folder) else []
                if csv_files:
                    c_selected_csvs = st.multiselect("Select Competition CSV(s)", csv_files, key="comp_csv_sel")
                    c_csv_paths = [os.path.join(csv_folder, f) for f in c_selected_csvs]
                else:
                    st.warning("No CSV files found in 'data/historical'.")
            with col_c_per:
                c_csv_interval_options = {
                    "1 Minute (1m)": "1m",
                    "5 Minutes (5m)": "5m",
                    "15 Minutes (15m)": "15m",
                    "30 Minutes (30m)": "30m",
                    "1 Hour (1h)": "1h",
                    "1 Day (1d)": "1d"
                }
                c_csv_int_label = st.selectbox("CSV Timeframe", list(c_csv_interval_options.keys()), key="comp_csv_tf")
                c_interval = c_csv_interval_options[c_csv_int_label]
            
        c_sentiment_csv_paths = []
        with st.expander("Optional: Competition News Sentiment"):
            news_folder = "data/news"
            news_files = [f for f in os.listdir(news_folder) if f.endswith('.csv')] if os.path.exists(news_folder) else []
            if news_files:
                c_selected_news = st.multiselect("Select Competition Sentiment CSV(s)", news_files)
                c_sentiment_csv_paths = [os.path.join(news_folder, f) for f in c_selected_news]
            else:
                st.info("No news CSV files found in 'data/news'.")

        if st.button("🏆 Start Competition"):
            if not selected_competitors:
                st.error("Select at least one competitor.")
            elif c_use_csv and not c_csv_paths:
                st.error("Please select at least one CSV file.")
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
                    status_text=status_text,
                    use_csv=c_use_csv,
                    csv_paths=c_csv_paths,
                    sentiment_csv_paths=c_sentiment_csv_paths
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
        # Header with Bulk Action
        col_hdr1, col_hdr2 = st.columns([4, 1])
        with col_hdr2:
            if st.button("🧨 DELETE ALL AGENTS", type="primary", use_container_width=True):
                for a in agents:
                    model_path = f"models/{a.name}.zip"
                    if os.path.exists(model_path): os.remove(model_path)
                    session.delete(a)
                session.commit()
                st.success("Wiped all agents!")
                time.sleep(1)
                st.rerun()

        # Custom Table with Delete Buttons
        st.write("---")
        # st.table is static, let's use columns for a more interactive list
        hdr_cols = st.columns([0.5, 2, 1, 1, 3, 0.5])
        hdr_cols[0].write("**ID**")
        hdr_cols[1].write("**Name**")
        hdr_cols[2].write("**Score 🏆**")
        hdr_cols[3].write("**Balance ($)**")
        hdr_cols[4].write("**Strategy**")
        hdr_cols[5].write("**Del**")

        for a in agents:
            row_cols = st.columns([0.5, 2, 1, 1, 3, 0.5])
            row_cols[0].write(a.id)
            row_cols[1].write(a.name)
            row_cols[2].write(int(a.score))
            row_cols[3].write(f"{a.balance:,.2f}")
            row_cols[4].write(a.strategy_type)
            if row_cols[5].button("🗑️", key=f"del_{a.id}"):
                model_path = f"models/{a.name}.zip"
                if os.path.exists(model_path): os.remove(model_path)
                session.delete(a)
                session.commit()
                st.rerun()
    else:
        st.write("No agents in the arena yet.")

with tab_pos:
    st.subheader("Current Open Positions Summary")
    
    positions = session.query(OpenPosition).all()
    
    if positions:
        # Performance optimization: Summarize first
        total_trades = len(positions)
        
        # Calculate PnL for all positions
        with st.spinner("Calculating total PnL..."):
            total_unrealized_pnl = 0.0
            latest_prices = fetcher.fetch_current_prices()
            
            table_data = []
            for p in positions:
                current_price = latest_prices.get(p.symbol)
                pnl = 0.0
                if current_price:
                    pnl = engine.calculate_pnl(p.position_type, p.lots, p.open_price, current_price)
                    pnl -= p.brokerage_fee
                
                total_unrealized_pnl += pnl
                table_data.append({
                    "Agent": p.agent.name,
                    "Symbol": p.symbol,
                    "Type": p.position_type,
                    "Lots": p.lots,
                    "Open Price": p.open_price,
                    "Current Price": f"{current_price:.5f}" if current_price else "N/A",
                    "Net PnL ($)": round(pnl, 2)
                })

        # Display Metrics
        m1, m2 = st.columns(2)
        m1.metric("Total Open Trades", total_trades)
        m2.metric("Total Unrealized PnL", f"${total_unrealized_pnl:,.2f}", delta=f"{total_unrealized_pnl:.2f}")
        
        st.divider()
        if st.checkbox("Show Detailed Position List"):
            st.table(table_data)
            
        if st.button("🔄 Refresh Data (Manual)"):
            st.rerun()
    else:
        st.info("No open positions at the moment.")
        if st.button("🔄 Refresh"):
            st.rerun()
