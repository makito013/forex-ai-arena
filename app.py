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


def main():
    """Streamlit app entrypoint.
    Wrapped in a function so that multiprocessing workers (spawned on macOS for
    concurrent training) can re-import this module without executing the UI.
    """

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

    # Session State Initialization for Goal-Oriented Mode
    if "goal_training_active" not in st.session_state:
        st.session_state.goal_training_active = False
        st.session_state.goal_paused = False
        st.session_state.goal_epoch = 1
        st.session_state.goal_best_pnl = -99999.0
        st.session_state.goal_best_dd = 0.0

    # Setup Tabs
    tab_setup, tab_board, tab_comp, tab_train, tab_data, tab_pos = st.tabs([
        "Arena Setup (Math Test)", "Leaderboard", "Competition Arena ⚔️", "Training Arena 🧠", "Market Data", "Open Positions"
    ])

    with tab_train:
        st.header("🧠 Train New AI Agents")
    
        if st.session_state.goal_training_active:
            st.warning("🎯 **Goal-Oriented Training is currently running in the background.**")
        
            # Display Controls
            col_p, col_save, col_s = st.columns(3)
            if col_p.button("⏸️ Pause / Resume"):
                st.session_state.goal_paused = not st.session_state.goal_paused
                st.rerun()

            if col_save.button("💾 Stop & Save Agents", help="Stops training and saves the current population to the arena. You can then pick them as Base Agents to continue training with different parameters (leverage, drawdown, goal)."):
                last_results = {r[0]: r for r in st.session_state.get("goal_last_results", [])}
                saved_count = 0
                for a_name in st.session_state.goal_agents:
                    if not os.path.exists(f"models/{a_name}.zip"):
                        continue
                    if session.query(Agent).filter_by(name=a_name).first():
                        continue
                    res = last_results.get(a_name)
                    a_bal = res[1] if res else engine.config['arena']['initial_balance']
                    a_score = res[2] if res else 0.0
                    session.add(Agent(name=a_name, strategy_type="GOAL_SAVED_FREE_MTF", balance=a_bal, score=a_score))
                    saved_count += 1
                session.commit()
                st.session_state.goal_training_active = False
                st.session_state.goal_paused = False
                st.session_state.goal_saved_msg = f"💾 Saved {saved_count} agent(s) to the arena. Select them as **Base Agents** in Deep Evolutionary or Goal-Oriented mode to continue their training with new parameters."
                st.rerun()

            if col_s.button("🛑 Stop & Reset"):
                st.session_state.goal_training_active = False
                st.session_state.goal_paused = False
                st.rerun()
            
            goal_is_monthly = st.session_state.get("goal_type", "Total Net Profit ($)").startswith("Monthly")
            st.metric("Target Monthly Profit ($/month)" if goal_is_monthly else "Target Net Profit ($)",
                      f"${st.session_state.target_profit_goal:,.2f}")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Current Epoch", st.session_state.goal_epoch)
            m_col2.metric("Best Monthly PnL So Far" if goal_is_monthly else "Best Agent PnL So Far",
                          f"${st.session_state.goal_best_pnl:,.2f}")
            m_col3.metric("Max Drawdown (Best Agent)", f"{st.session_state.get('goal_best_dd', 0.0) * 100:.1f}%",
                          help="Worst peak-to-valley equity loss of the best agent. Winners must stay under the allowed limit.")
        
            if not st.session_state.goal_paused:
                with st.spinner(f"🚀 Epoch {st.session_state.goal_epoch}: {len(st.session_state.goal_agents)} agents exploring random pairs/timeframes concurrently. Please wait..."):
                    from src.engine.trainer import run_concurrent_epoch_free
                    results = run_concurrent_epoch_free(
                        st.session_state.goal_agents,
                        st.session_state.get("goal_config", engine.config),
                        max_steps_per_epoch=st.session_state.get("goal_max_steps", 20000),
                        sentiment_csv_paths=st.session_state.get("goal_sentiment_paths")
                    )

                    if not results:
                        st.session_state.goal_training_active = False
                        st.error("Epoch produced no results — no readable CSV datasets found in 'data/historical'. Training stopped.")
                        st.stop()

                    # Keep the latest results so 'Stop & Save' can persist real balances/scores
                    st.session_state.goal_last_results = results

                    st.session_state.goal_epoch += 1
                
                    # Check Results & Evolutionary Crossover
                    initial_bal = engine.config['arena']['initial_balance']
                    max_dd_allowed = st.session_state.get("goal_max_dd", 30.0)
                    limit_frac = max_dd_allowed / 100.0
                    target = st.session_state.target_profit_goal

                    def _goal_metric(bal, days):
                        pnl = bal - initial_bal
                        return pnl * (30.0 / max(days, 1.0)) if goal_is_monthly else pnl

                    def _fitness(r):
                        # Risk-adjusted: profit minus a penalty for drawdown beyond the limit.
                        # At 100% drawdown the penalty costs ~one full target, so a high-DD
                        # gambler is ranked below a steadier, lower-DD agent.
                        _, bal, _, dd, days = r
                        excess = max(0.0, dd - limit_frac)
                        penalty = (excess / max(1e-9, 1.0 - limit_frac)) * abs(target)
                        return _goal_metric(bal, days) - penalty

                    # Rank by risk-adjusted fitness so cloning promotes LOW-drawdown agents
                    results.sort(key=_fitness, reverse=True)

                    # Best-of-epoch follows the elite that is actually being promoted
                    best_name, best_bal, best_score, best_dd_this_epoch, best_days = results[0]
                    best_pnl_this_epoch = _goal_metric(best_bal, best_days)

                    # Winner = hit the (monthly or total) profit target WITHOUT a giant drawdown.
                    # Monthly goal also requires a window of at least 7 days — one lucky day
                    # extrapolated x30 is not a sustainable monthly profit.
                    winners = []
                    for name, bal, score, dd, days in results:
                        goal_metric = _goal_metric(bal, days)
                        enough_history = (not goal_is_monthly) or days >= 7.0
                        if goal_metric >= target and dd * 100 <= max_dd_allowed and enough_history:
                            winners.append((name, bal, score))

                    if best_pnl_this_epoch > st.session_state.goal_best_pnl:
                        st.session_state.goal_best_pnl = best_pnl_this_epoch
                        st.session_state.goal_best_dd = best_dd_this_epoch

                    if winners:
                        goal_unit = "/month" if goal_is_monthly else ""
                        st.success(f"🎉 Target Reached! {len(winners)} agents hit the goal of ${st.session_state.target_profit_goal}{goal_unit} while keeping drawdown under {max_dd_allowed:.0f}%.")
                        # Save Winners to DB
                        for w_name, w_bal, w_score in winners:
                            new_agent = Agent(name=w_name, strategy_type=f"GOAL_{st.session_state.goal_symbol}", balance=w_bal, score=w_score)
                            session.add(new_agent)
                        session.commit()
                        st.session_state.goal_training_active = False
                        st.info("Check the Leaderboard to see your new profitable agents!")
                    else:
                        # Evolutionary step: Clone the top 20% to replace the bottom 20%
                        num_agents_run = len(results)
                        if num_agents_run >= 5:
                            elite_count = max(1, num_agents_run // 5)
                            elites = results[:elite_count]
                            losers = results[-elite_count:]
                        
                            import shutil
                            for i in range(elite_count):
                                elite_name = elites[i][0]
                                loser_name = losers[i][0]
                                elite_path = f"models/{elite_name}.zip"
                                loser_path = f"models/{loser_name}.zip"
                                if os.path.exists(elite_path):
                                    shutil.copy(elite_path, loser_path)
                                
                        # Loop again
                        st.rerun()
            else:
                st.info("⏸️ Training Paused. Click 'Pause / Resume' to continue.")

        else:
            if st.session_state.get("goal_saved_msg"):
                st.success(st.session_state.pop("goal_saved_msg"))

            st.write("Configure and launch a training session directly from the UI. The agents will play against historical data for the chosen period and asset.")
        
            mode = st.radio("Training Mode", ["Create New Agent(s)", "Retrain Existing Agent(s)", "Deep Evolutionary Training 🧬", "Goal-Oriented Concurrent 🎯"], horizontal=True)
            free_exploration = mode in ("Deep Evolutionary Training 🧬", "Goal-Oriented Concurrent 🎯")

            selected_symbol = "CSV"
            selected_period = "max"
            selected_interval = "15m"
            use_csv = False
            csv_paths = []
            sentiment_csv_paths = []
            existing_agent_names = None
            num_agents = 1
            target_epochs = 10
            target_profit_goal = 5000.0
            goal_type = "Total Net Profit ($)"
            max_steps_per_epoch = 20000
            max_dd_allowed = 30.0
            base_agent_names = []
            leverage_override = int(engine.config['trading']['leverage'])
            available_csvs = []

            if free_exploration:
                csv_folder = "data/historical"
                available_csvs = [f for f in os.listdir(csv_folder) if f.endswith('.csv')] if os.path.exists(csv_folder) else []

                if mode == "Deep Evolutionary Training 🧬":
                    st.info("🧬 **Total Freedom Mode**: Survival of the fittest. Each agent freely explores ALL pairs and timeframes found in 'data/historical' — a random dataset (pair + timeframe) is picked at every epoch. No symbol or timeframe selection needed; the agent hunts for profit wherever it finds it. Only profitable agents are saved.")
                    st.caption(f"📂 {len(available_csvs)} dataset(s) available in '{csv_folder}' for free exploration.")

                    col_pop, col_ep, col_ms = st.columns(3)
                    with col_pop:
                        num_agents = st.number_input("Population Size", min_value=1, max_value=50, value=5)
                    with col_ep:
                        target_epochs = st.slider("Exploration Epochs (random pair/timeframe each epoch)", min_value=1, max_value=100, value=20)
                    with col_ms:
                        max_steps_per_epoch = st.number_input("Max Candles per Epoch", min_value=1000, max_value=200000, value=20000, step=1000, help="Large datasets (e.g. M1) are sliced to a random window of this size each epoch, so every epoch takes a similar amount of time regardless of timeframe.")

                    col_base, col_lev = st.columns([2, 1])
                    with col_base:
                        all_agent_names = [a.name for a in session.query(Agent).all()]
                        base_agent_names = st.multiselect("Base Agents (optional): start from clones of saved agents", all_agent_names, key="evo_base", help="Each new agent starts from a copy of one of these brains (round-robin) instead of zero — continue training with new parameters.")
                    with col_lev:
                        leverage_override = st.number_input("Leverage X", min_value=1, max_value=2000, value=int(engine.config['trading']['leverage']), key="evo_lev", help="Applied to THIS training session only. Lower leverage = more margin per lot AND a bigger brokerage fee (fee is 5% of margin), making profit harder but more realistic.")
                else:
                    st.info("🎯 **Goal-Oriented with Total Freedom**: Agents train concurrently, each one freely exploring ALL pairs and timeframes in 'data/historical' — a random dataset + random window every epoch — until one reaches the target profit. No symbol or timeframe selection needed.")
                    st.caption(f"📂 {len(available_csvs)} dataset(s) available in '{csv_folder}' for free exploration.")

                    goal_type = st.radio("Goal Type", ["Total Net Profit ($)", "Monthly Profit ($/month)"], horizontal=True,
                                         help="Monthly Profit normalizes each agent's result by the time span of the data it traded: profit ÷ days × 30. The agent wins when it sustains the requested profit PER MONTH (e.g. $700 in 14 days of M1 data = $1,500/month).")

                    col_na, col_tp, col_dd, col_ms = st.columns(4)
                    with col_na:
                        num_agents = st.number_input("Concurrent Agents", min_value=1, max_value=20, value=5)
                    with col_tp:
                        if goal_type == "Monthly Profit ($/month)":
                            target_profit_goal = st.number_input("Target Monthly Profit ($/month)", min_value=100.0, value=1000.0, step=100.0)
                        else:
                            target_profit_goal = st.number_input("Target Net Profit ($)", min_value=100.0, value=5000.0, step=500.0)
                    with col_dd:
                        max_dd_allowed = st.number_input("Max Drawdown Allowed (%)", min_value=5.0, max_value=100.0, value=30.0, step=5.0, help="A winner must hit the profit target WITHOUT ever losing more than this % from its equity peak. Prevents 'lucky gambler' agents that nearly zeroed the account on the way up.")
                    with col_ms:
                        max_steps_per_epoch = st.number_input("Max Candles per Epoch", min_value=1000, max_value=200000, value=20000, step=1000, help="Large datasets (e.g. M1) are sliced to a random window of this size each epoch, so every epoch takes a similar amount of time regardless of timeframe.")

                    col_base, col_lev = st.columns([2, 1])
                    with col_base:
                        all_agent_names = [a.name for a in session.query(Agent).all()]
                        base_agent_names = st.multiselect("Base Agents (optional): continue from clones of saved agents", all_agent_names, key="goal_base", help="The population starts from copies of these brains (round-robin) instead of zero. Use after '💾 Stop & Save' to continue training with new parameters (leverage, drawdown, goal).")
                    with col_lev:
                        leverage_override = st.number_input("Leverage X", min_value=1, max_value=2000, value=int(engine.config['trading']['leverage']), key="goal_lev", help="Applied to THIS training session only. Lower leverage = more margin per lot AND a bigger brokerage fee (fee is 5% of margin), making profit harder but more realistic.")
            else:
                t_data_source = st.radio("Data Source", ["Yahoo Finance (Online)", "Local CSV (Offline)"], key="train_ds")

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
                        selected_period_label = st.selectbox("Base Execution Timeframe", list(period_options.keys()), help="The speed at which the agent makes decisions. H4 and D1 are automatically calculated in the background.")
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
                        csv_int_label = st.selectbox("Base Execution Timeframe", list(csv_interval_options.keys()), help="The frequency of agent actions. MTF indicators (H4, D1) are generated automatically.")
                        selected_interval = csv_interval_options[csv_int_label]

                with col_ag:
                    if mode == "Retrain Existing Agent(s)":
                        existing_agents = session.query(Agent).all()
                        if existing_agents:
                            agent_names = [a.name for a in existing_agents]
                            existing_agent_names = st.multiselect("Select Agent(s) to Retrain", agent_names, default=None)
                        else:
                            st.warning("No existing agents found in database.")
                    else:
                        num_agents = st.number_input("Number of New Agents to Train", min_value=1, max_value=10, value=1)

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

            if st.button("🚀 Start Training Session"):
                if mode == "Retrain Existing Agent(s)" and not existing_agent_names:
                    st.error("Please select at least one agent to retrain.")
                elif free_exploration and not available_csvs:
                    st.error("No CSV files found in 'data/historical'. Add historical data files first.")
                elif not free_exploration and use_csv and not csv_paths:
                    st.error("Please place a CSV file in 'data/historical' and select it.")
                else:
                    st.divider()
                
                    if mode == "Goal-Oriented Concurrent 🎯":
                        # Initialize Goal Training State (Total Freedom: each agent picks a
                        # random dataset + window per epoch, so no upfront data loading)
                        from src.engine.trainer import generate_agent_name
                        import copy, shutil

                        # Population: fresh brains, or clones of the selected base agents (round-robin)
                        population = []
                        for p_idx in range(num_agents):
                            clone_name = generate_agent_name()
                            if base_agent_names:
                                base_src = f"models/{base_agent_names[p_idx % len(base_agent_names)]}.zip"
                                if os.path.exists(base_src):
                                    shutil.copy(base_src, f"models/{clone_name}.zip")
                            population.append(clone_name)

                        # Per-session config override (leverage + drawdown limit for this run)
                        goal_config = copy.deepcopy(engine.config)
                        goal_config['trading']['leverage'] = int(leverage_override)
                        goal_config['trading']['max_drawdown_limit'] = max_dd_allowed / 100.0
                        st.session_state.goal_config = goal_config

                        st.session_state.goal_agents = population
                        st.session_state.goal_symbol = "FREE_MTF"
                        st.session_state.goal_max_steps = max_steps_per_epoch
                        st.session_state.goal_max_dd = max_dd_allowed
                        st.session_state.goal_type = goal_type
                        st.session_state.goal_sentiment_paths = sentiment_csv_paths
                        st.session_state.target_profit_goal = target_profit_goal

                        # Start Loop
                        st.session_state.goal_training_active = True
                        st.session_state.goal_epoch = 1
                        st.session_state.goal_best_pnl = -99999.0
                        st.session_state.goal_best_dd = 0.0
                        st.rerun()
                    else:
                        # Standard Training Flow
                        overall_status = st.empty()
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                    
                        st.write("---")
                        st.subheader("📊 Live Training Monitor")
                        mon_col1, mon_col2 = st.columns(2)
                        trade_metric = mon_col1.empty()
                        pnl_metric = mon_col2.empty()
                    
                        if mode == "Deep Evolutionary Training 🧬":
                            from src.engine.trainer import run_free_exploration_training
                            import copy
                            evo_config = copy.deepcopy(engine.config)
                            evo_config['trading']['leverage'] = int(leverage_override)
                            evo_config['trading']['max_drawdown_limit'] = max_dd_allowed / 100.0
                            success, results = run_free_exploration_training(
                                num_agents=num_agents,
                                config=evo_config,
                                progress_bar=progress_bar,
                                status_text=status_text,
                                overall_status=overall_status,
                                trade_metric=trade_metric,
                                pnl_metric=pnl_metric,
                                sentiment_csv_paths=sentiment_csv_paths,
                                target_epochs=target_epochs,
                                max_steps_per_epoch=max_steps_per_epoch,
                                base_agent_names=base_agent_names
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

            # Match the conditions the agent trained under (avoids unfair train↔arena mismatch)
            st.markdown("**⚙️ Match Training Conditions** — set these to the SAME values used when the agent trained, or the result won't be comparable.")
            col_c_lev, col_c_win, col_c_det = st.columns(3)
            with col_c_lev:
                c_leverage = st.number_input("Leverage (1:N)", min_value=1, max_value=2000, value=int(engine.config['trading']['leverage']), key="comp_lev", help="The Arena uses config.yaml's leverage by default. If the agent trained with a different leverage, set it here — otherwise the position economics differ and it may blow up.")
            with col_c_win:
                c_max_window = st.number_input("Max Candles (0 = full dataset)", min_value=0, max_value=500000, value=20000, step=1000, key="comp_win", help="Caps the test to the most recent N candles, matching the training window size. The full dataset is a much longer continuous run — a high-drawdown agent survives a 20k window but blows up over 100k.")
            with col_c_det:
                c_deterministic = st.checkbox("Deterministic actions", value=True, key="comp_det", help="PPO trains by sampling actions stochastically. Forcing deterministic can collapse the policy. Uncheck to replay the agent the way it learned.")

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
                        sentiment_csv_paths=c_sentiment_csv_paths,
                        leverage_override=c_leverage,
                        max_window=(c_max_window if c_max_window > 0 else None),
                        deterministic=c_deterministic
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
                        pnl = engine.calculate_pnl(p.position_type, p.lots, p.open_price, current_price, contract_size=engine.get_contract_size(p.symbol), is_base_usd=engine.is_usd_base(p.symbol))
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


if __name__ == "__main__":
    main()
