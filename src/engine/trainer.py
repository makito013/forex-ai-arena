import os
import string
import random
import numpy as np
import pandas as pd
import concurrent.futures
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.data.fetcher import MarketDataFetcher
from src.engine.financial import FinancialEngine
from src.engine.env import ForexEnv
from src.db_models import init_db, Agent

# --- Concurrent Worker Function ---
def train_single_agent_chunk(args):
    """
    Worker function for training a single agent in a separate process.
    """
    agent_name, df, config, steps_per_day, timesteps = args
    from stable_baselines3 import PPO
    from src.engine.financial import FinancialEngine
    from src.engine.env import ForexEnv
    import os
    
    # Re-initialize engine locally for the process
    engine = FinancialEngine('config.yaml')
    engine.config = config 
    
    env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=steps_per_day)
    
    model_path = f"models/{agent_name}.zip"
    if os.path.exists(model_path):
        try:
            model = PPO.load(model_path, env=env)
        except:
            model = PPO("MlpPolicy", env, verbose=0)
    else:
        model = PPO("MlpPolicy", env, verbose=0)
        
    # Learn for exactly 1 epoch (length of the dataframe)
    model.learn(total_timesteps=timesteps)
    
    os.makedirs('models', exist_ok=True)
    model.save(f"models/{agent_name}")
    
    return agent_name, env.balance, env.score

def run_concurrent_epoch(agent_names, df, config, steps_per_day):
    """
    Orchestrates the training of multiple agents concurrently for 1 epoch.
    """
    timesteps = len(df)
    results = []
    
    # Determine optimal worker count
    max_workers = min(len(agent_names), os.cpu_count() or 4)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for name in agent_names:
            args = (name, df, config, steps_per_day, timesteps)
            futures.append(executor.submit(train_single_agent_chunk, args))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Concurrent Worker Error: {e}")
                
    return results

class StreamlitCallback(BaseCallback):
    """
    Custom callback for reporting training progress and real-time stats to the Streamlit UI.
    """
    def __init__(self, progress_bar, status_text, total_timesteps, agent_name, trade_metric=None, pnl_metric=None, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        self.trade_metric = trade_metric
        self.pnl_metric = pnl_metric

    def _on_step(self) -> bool:
        # Throttle UI updates to avoid freezing Streamlit (every 100 steps)
        if self.num_timesteps % 100 == 0 or self.num_timesteps == self.total_timesteps:
            progress = self.num_timesteps / self.total_timesteps
            safe_progress = max(0.0, min(progress, 1.0))
            self.progress_bar.progress(safe_progress)
            
            # Get data from the environment
            env_info = self.training_env.get_attr('balance')[0]
            initial_balance = self.training_env.get_attr('initial_balance')[0]
            current_pnl = env_info - initial_balance
            has_pos = self.training_env.get_attr('current_position')[0] is not None
            
            self.status_text.text(f"Training {self.agent_name} | Step: {self.num_timesteps}/{self.total_timesteps} | PnL: ${current_pnl:.2f}")
            
            # Update live stats in place using .empty() placeholders
            if self.trade_metric and self.pnl_metric:
                self.trade_metric.metric("Active Trade", "YES" if has_pos else "NO")
                # Format color: red if negative, green if positive
                self.pnl_metric.metric("Current Agent PnL", f"${current_pnl:.2f}", delta=f"{current_pnl:.2f}")
            
        return True

def generate_agent_name():
    return "Agent_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

def run_training_session(symbol, period, interval, num_agents, config, progress_bar, status_text, overall_status, trade_metric=None, pnl_metric=None, existing_agent_names=None, use_csv=False, csv_paths=None, sentiment_csv_paths=None):
    engine = FinancialEngine('config.yaml')
    fetcher = MarketDataFetcher(config)
    session = init_db()
    
    if use_csv and csv_paths:
        overall_status.info(f"Loading data from {len(csv_paths)} CSV file(s)...")
        df = fetcher.fetch_from_multiple_csvs(csv_paths)
        symbol_label = "+".join([os.path.basename(p) for p in csv_paths])
    else:
        overall_status.info(f"Fetching historical data for {symbol} ({period} / {interval})...")
        df = fetcher.fetch_historical_data(symbol, period=period, interval=interval)
        symbol_label = symbol
    
    if df.empty:
        overall_status.error(f"Failed to fetch data. Training aborted.")
        return False, []

    # Optional Sentiment Integration
    if sentiment_csv_paths:
        overall_status.info(f"Integrating sentiment data from {len(sentiment_csv_paths)} file(s)...")
        sentiment_df = fetcher.load_sentiment_from_multiple_csvs(sentiment_csv_paths)
        df = fetcher.attach_sentiment_to_df(df, sentiment_df)
        overall_status.success("Sentiment integrated into market data.")

    data_length = len(df)
    
    if existing_agent_names:
        num_agents = len(existing_agent_names)
        overall_status.success(f"Data loaded! {data_length} candles. Continuing training for {num_agents} existing agent(s).")
    else:
        overall_status.success(f"Data loaded! {data_length} candles available. Training {num_agents} new agent(s).")
    
    results = []
    
    for i in range(num_agents):
        if existing_agent_names:
            agent_name = existing_agent_names[i]
            db_agent = session.query(Agent).filter_by(name=agent_name).first()
            initial_balance = db_agent.balance if db_agent else config['arena']['initial_balance']
        else:
            agent_name = generate_agent_name()
            initial_balance = config['arena']['initial_balance']
            db_agent = None
            
        overall_status.info(f"Initializing Environment for Agent {i+1}/{num_agents}: {agent_name}...")
        
        mapping = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}
        steps_per_day = mapping.get(interval, 96)
        
        env = ForexEnv(df=df, engine=engine, initial_balance=initial_balance, steps_per_day=steps_per_day)
        
        model_path = f"models/{agent_name}.zip"
        if os.path.exists(model_path):
            overall_status.info(f"Loading existing model weights for {agent_name}...")
            try:
                # Compatibility check: see if the loaded model expects 7 or 8 variables
                temp_model = PPO.load(model_path)
                model_obs_shape = temp_model.observation_space.shape[0]
                
                if model_obs_shape != env.observation_space.shape[0]:
                    from gymnasium import spaces
                    original_obs_func = env._next_observation
                    
                    if model_obs_shape < 11: # Legacy (7 or 8)
                        overall_status.warning(f"Agent {agent_name} is legacy ({model_obs_shape} obs). Adjusting environment...")
                        env.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(model_obs_shape,), dtype=np.float32)
                        def legacy_obs():
                            full_obs = original_obs_func()
                            return full_obs[:model_obs_shape]
                        env._next_observation = legacy_obs
                
                model = PPO.load(model_path, env=env)
            except Exception as e:
                overall_status.error(f"Error loading {agent_name}: {e}. Starting fresh.")
                model = PPO("MlpPolicy", env, verbose=0)
        else:
            model = PPO("MlpPolicy", env, verbose=0)
        
        timesteps = min(50000, data_length * 3)
        callback = StreamlitCallback(progress_bar, status_text, timesteps, agent_name, trade_metric, pnl_metric)
        model.learn(total_timesteps=timesteps, callback=callback)
        
        final_balance = env.balance
        strategy_label = f"PPO_{symbol_label}" if use_csv else f"PPO_{symbol}_{period}"
        
        if db_agent:
            db_agent.balance = final_balance
            if symbol_label not in db_agent.strategy_type:
                db_agent.strategy_type += f" + {symbol_label}"
            strategy_label = db_agent.strategy_type
        else:
            new_agent = Agent(name=agent_name, strategy_type=strategy_label, balance=final_balance)
            session.add(new_agent)
            
        session.commit()
        
        os.makedirs('models', exist_ok=True)
        model.save(f"models/{agent_name}")
        
        results.append({
            "Agent": agent_name,
            "Symbol": symbol_label,
            "Strategy": strategy_label,
            "Final Balance": round(final_balance, 2)
        })
        
    return True, results

def run_deep_evolutionary_training(symbol, period, interval, num_agents, config, progress_bar, status_text, overall_status, trade_metric=None, pnl_metric=None, use_csv=False, csv_paths=None, sentiment_csv_paths=None, target_epochs=20):
    """
    Automated training pipeline:
    1. Trains multiple agents.
    2. Runs for many epochs.
    3. Only 'survivors' (those with positive balance) are saved.
    """
    engine = FinancialEngine('config.yaml')
    fetcher = MarketDataFetcher(config)
    session = init_db()
    
    if use_csv and csv_paths:
        overall_status.info(f"Loading data from {len(csv_paths)} CSV file(s)...")
        df = fetcher.fetch_from_multiple_csvs(csv_paths)
        symbol_label = "+".join([os.path.basename(p) for p in csv_paths])
    else:
        overall_status.info(f"Fetching data for {symbol}...")
        df = fetcher.fetch_historical_data(symbol, period=period, interval=interval)
        symbol_label = symbol
    
    if df.empty:
        overall_status.error("Failed to load data.")
        return False, []

    if sentiment_csv_paths:
        overall_status.info(f"Integrating sentiment from {len(sentiment_csv_paths)} file(s)...")
        sentiment_df = fetcher.load_sentiment_from_multiple_csvs(sentiment_csv_paths)
        df = fetcher.attach_sentiment_to_df(df, sentiment_df)

    data_length = len(df)
    # Deep training: more steps!
    timesteps_per_agent = data_length * target_epochs
    
    overall_status.info(f"Starting Deep Training session. Target: {num_agents} agents, {target_epochs} passes each ({timesteps_per_agent} steps total). Only profitable agents will be kept.")
    
    survivors = []
    
    for i in range(num_agents):
        agent_name = generate_agent_name()
        overall_status.info(f"Evolution Phase {i+1}/{num_agents}: Training {agent_name}...")
        
        mapping = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}
        steps_per_day = mapping.get(interval, 96)
        
        env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=steps_per_day)
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = StreamlitCallback(progress_bar, status_text, timesteps_per_agent, agent_name, trade_metric, pnl_metric)
        model.learn(total_timesteps=timesteps_per_agent, callback=callback)
        
        # Validation Check: Did it actually make profit?
        final_balance = env.balance
        initial_balance = config['arena']['initial_balance']
        
        if final_balance > initial_balance:
            overall_status.success(f"📈 Survival! {agent_name} ended with ${final_balance:.2f}. Saving...")
            
            strategy_label = f"EVO_{symbol_label}"
            new_agent = Agent(name=agent_name, strategy_type=strategy_label, balance=final_balance, score=env.score)
            session.add(new_agent)
            session.commit()
            
            os.makedirs('models', exist_ok=True)
            model.save(f"models/{agent_name}")
            
            survivors.append({
                "Agent": agent_name,
                "Final Balance": round(final_balance, 2),
                "Profit": round(final_balance - initial_balance, 2),
                "Score": int(env.score)
            })
        else:
            overall_status.warning(f"💀 Extinction: {agent_name} failed to profit (${final_balance:.2f}). Discarding.")
            
    return True, survivors
