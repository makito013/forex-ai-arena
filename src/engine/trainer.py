import os
import re
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
    agent_name, df, config, steps_per_day, timesteps, symbol = args
    from stable_baselines3 import PPO
    from src.engine.financial import FinancialEngine
    from src.engine.env import ForexEnv
    import os

    # Re-initialize engine locally for the process (apply_config refreshes leverage/fees too)
    engine = FinancialEngine('config.yaml')
    engine.apply_config(config)

    env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=steps_per_day, symbol=symbol)
    
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

    # Calendar days spanned by the training window (for monthly PnL normalization)
    try:
        days_spanned = max(1.0, (df.index[-1] - df.index[0]).total_seconds() / 86400.0)
    except Exception:
        days_spanned = max(1.0, timesteps / max(steps_per_day, 1))

    return agent_name, env.balance, env.score, env.max_drawdown, days_spanned

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
            args = (name, df, config, steps_per_day, timesteps, None)
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
    def __init__(self, progress_bar, status_text, total_timesteps, agent_name, trade_metric=None, pnl_metric=None, verbose=0, progress_offset=0.0, progress_scale=1.0):
        super(StreamlitCallback, self).__init__(verbose)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        self.trade_metric = trade_metric
        self.pnl_metric = pnl_metric
        # Allows mapping this learn() call to a slice of the overall progress bar
        self.progress_offset = progress_offset
        self.progress_scale = progress_scale

    def _on_step(self) -> bool:
        # Throttle UI updates to avoid freezing Streamlit (every 100 steps)
        if self.num_timesteps % 100 == 0 or self.num_timesteps == self.total_timesteps:
            progress = self.progress_offset + (self.num_timesteps / self.total_timesteps) * self.progress_scale
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

# --- Free Exploration (Total Freedom) helpers ---
# Matches files like EURUSDM15.csv, XAUUSDH1.csv -> (symbol, timeframe)
TIMEFRAME_FILE_PATTERN = re.compile(r'^([A-Z]+?)(M1|M5|M15|M30|H1|H4|D1)\.csv$', re.IGNORECASE)
SUFFIX_TO_INTERVAL = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
INTERVAL_TO_STEPS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}

def discover_historical_datasets(folder="data/historical"):
    """
    Scans the historical data folder and infers (symbol, timeframe) from filenames.
    Files that don't match the naming convention default to a 15m interval.
    """
    datasets = []
    if not os.path.isdir(folder):
        return datasets
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.csv'):
            continue
        match = TIMEFRAME_FILE_PATTERN.match(fname)
        if match:
            symbol = match.group(1).upper()
            interval = SUFFIX_TO_INTERVAL[match.group(2).upper()]
        else:
            symbol = os.path.splitext(fname)[0]
            interval = "15m"
        datasets.append({"path": os.path.join(folder, fname), "symbol": symbol, "interval": interval})
    return datasets

# Module-level cache of indicator-prepared DataFrames (persists across Streamlit reruns)
_PREPARED_DATASET_CACHE = {}

def load_prepared_dataset(fetcher, path, sentiment_df=None):
    """
    Loads a CSV, attaches sentiment, prepares indicators over the FULL history and caches it.
    Returns None if the file is unreadable.
    """
    df = _PREPARED_DATASET_CACHE.get(path)
    if df is None:
        df = fetcher.fetch_from_csv(path)
        if df.empty:
            return None
        if sentiment_df is not None and not sentiment_df.empty:
            df = fetcher.attach_sentiment_to_df(df, sentiment_df)
        df = ForexEnv._prepare_indicators(df)
        _PREPARED_DATASET_CACHE[path] = df
    return df

def slice_random_window(df, max_steps):
    """
    Slices a random contiguous window so a huge M1 file doesn't make one epoch take
    100x longer than an H1 epoch. Indicators were computed on the full history,
    so the window keeps them valid (no warm-up NaNs).
    """
    if len(df) > max_steps:
        start = random.randint(0, len(df) - max_steps)
        return df.iloc[start:start + max_steps].copy()
    return df

def run_concurrent_epoch_free(agent_names, config, max_steps_per_epoch=20000, sentiment_csv_paths=None, csv_folder="data/historical"):
    """
    Free-exploration version of run_concurrent_epoch (Goal-Oriented Total Freedom):
    each agent trains concurrently on a RANDOM dataset (any pair, any timeframe)
    and a random window picked for this epoch. Returns [(name, balance, score), ...].
    """
    fetcher = MarketDataFetcher(config)
    datasets = discover_historical_datasets(csv_folder)
    if not datasets:
        return []

    sentiment_df = None
    if sentiment_csv_paths:
        sentiment_df = fetcher.load_sentiment_from_multiple_csvs(sentiment_csv_paths)

    jobs = []
    for name in agent_names:
        window = None
        for _ in range(5):  # tolerate the occasional unreadable file
            ds = random.choice(datasets)
            df = load_prepared_dataset(fetcher, ds['path'], sentiment_df)
            if df is not None:
                window = slice_random_window(df, max_steps_per_epoch)
                break
        if window is None:
            continue
        steps_per_day = INTERVAL_TO_STEPS_PER_DAY.get(ds['interval'], 96)
        jobs.append((name, window, config, steps_per_day, len(window), ds['symbol']))

    if not jobs:
        return []

    results = []
    max_workers = min(len(jobs), os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_single_agent_chunk, args) for args in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Concurrent Worker Error: {e}")

    return results

def run_free_exploration_training(num_agents, config, progress_bar, status_text, overall_status, trade_metric=None, pnl_metric=None, sentiment_csv_paths=None, target_epochs=20, csv_folder="data/historical", max_steps_per_epoch=20000, base_agent_names=None):
    """
    Total Freedom evolutionary training:
    Each agent picks a RANDOM dataset (any pair, any timeframe) from data/historical at every
    epoch, learning across all available timeframes. Only profitable agents survive.

    Datasets larger than max_steps_per_epoch are sliced to a random contiguous window of that
    size each epoch, so an M1 epoch takes roughly as long as an H1 epoch (and each visit
    explores a different historical period).

    base_agent_names: optional list of saved agents — each new agent starts from a
    clone of one of them (round-robin) instead of a fresh brain, so training can
    continue with different parameters (leverage, drawdown limits, etc.).
    """
    engine = FinancialEngine('config.yaml')
    engine.apply_config(config)
    fetcher = MarketDataFetcher(config)
    session = init_db()

    datasets = discover_historical_datasets(csv_folder)
    if not datasets:
        overall_status.error(f"No CSV files found in '{csv_folder}'. Training aborted.")
        return False, []

    sentiment_df = None
    if sentiment_csv_paths:
        overall_status.info(f"Loading sentiment from {len(sentiment_csv_paths)} file(s)...")
        sentiment_df = fetcher.load_sentiment_from_multiple_csvs(sentiment_csv_paths)

    overall_status.info(f"🧬 Total Freedom: {num_agents} agents x {target_epochs} epochs. {len(datasets)} datasets available (all pairs & timeframes). Only profitable agents will be kept.")

    total_slices = num_agents * target_epochs
    survivors = []

    for i in range(num_agents):
        agent_name = generate_agent_name()
        model = None
        env = None
        explored = []
        worst_dd = 0.0

        for epoch in range(target_epochs):
            ds = random.choice(datasets)
            ds_label = f"{ds['symbol']} {ds['interval']}"
            explored.append(ds_label)
            overall_status.info(f"🧬 Agent {i+1}/{num_agents} ({agent_name}) | Epoch {epoch+1}/{target_epochs} | Exploring {ds_label}...")

            df = load_prepared_dataset(fetcher, ds['path'], sentiment_df)
            if df is None:
                overall_status.warning(f"Skipping unreadable dataset: {ds['path']}")
                continue

            # Balance epoch duration: random window keeps M1/H1 epochs comparable
            window_df = slice_random_window(df, max_steps_per_epoch)

            env = ForexEnv(df=window_df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=INTERVAL_TO_STEPS_PER_DAY.get(ds['interval'], 96), symbol=ds['symbol'])

            if model is None:
                # Continue from a base agent's brain when provided (round-robin)
                if base_agent_names:
                    base_name = base_agent_names[i % len(base_agent_names)]
                    base_path = f"models/{base_name}.zip"
                    if os.path.exists(base_path):
                        try:
                            model = PPO.load(base_path, env=env)
                            overall_status.info(f"🧬 {agent_name} continues from base agent {base_name}.")
                        except Exception:
                            overall_status.warning(f"Base agent {base_name} is incompatible (old observation space?). {agent_name} starts fresh.")
                if model is None:
                    model = PPO("MlpPolicy", env, verbose=0)
            else:
                model.set_env(env)

            timesteps = len(env.df)
            slice_index = i * target_epochs + epoch
            callback = StreamlitCallback(
                progress_bar, status_text, timesteps, f"{agent_name} [{ds_label}]",
                trade_metric, pnl_metric,
                progress_offset=slice_index / total_slices,
                progress_scale=1.0 / total_slices
            )
            model.learn(total_timesteps=timesteps, callback=callback)
            worst_dd = max(worst_dd, env.max_drawdown)

        if model is None or env is None:
            overall_status.warning(f"💀 {agent_name} had no valid data to train on. Discarding.")
            continue

        # Validation Check: Did it actually make profit?
        final_balance = env.balance
        initial_balance = config['arena']['initial_balance']

        if final_balance > initial_balance:
            overall_status.success(f"📈 Survival! {agent_name} ended with ${final_balance:.2f}. Saving...")

            timeframes_seen = sorted(set(label.split()[1] for label in explored))
            strategy_label = f"EVO_FREE[{','.join(timeframes_seen)}]"
            new_agent = Agent(name=agent_name, strategy_type=strategy_label, balance=final_balance, score=env.score)
            session.add(new_agent)
            session.commit()

            os.makedirs('models', exist_ok=True)
            model.save(f"models/{agent_name}")

            survivors.append({
                "Agent": agent_name,
                "Final Balance": round(final_balance, 2),
                "Profit": round(final_balance - initial_balance, 2),
                "Max Drawdown": f"{worst_dd * 100:.1f}%",
                "Score": int(env.score),
                "Datasets Explored": f"{len(set(explored))} ({', '.join(sorted(set(explored)))})"
            })
        else:
            overall_status.warning(f"💀 Extinction: {agent_name} failed to profit (${final_balance:.2f}). Discarding.")

    return True, survivors

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
        
        env = ForexEnv(df=df, engine=engine, initial_balance=initial_balance, steps_per_day=steps_per_day, symbol=symbol_label)

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
                    
                    if model_obs_shape < env.observation_space.shape[0]: # Legacy (7, 8, 11, 22, or 26)
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

def run_deep_evolutionary_training(symbol, period, interval, num_agents, config, progress_bar, status_text, overall_status, trade_metric=None, pnl_metric=None, use_csv=False, csv_paths=None, sentiment_csv_paths=None, target_epochs=20, free_exploration=False):
    """
    Automated training pipeline:
    1. Trains multiple agents.
    2. Runs for many epochs.
    3. Only 'survivors' (those with positive balance) are saved.

    With free_exploration=True the symbol/interval/csv selection is ignored: each agent
    freely picks a random dataset (any pair, any timeframe) from data/historical per epoch.
    """
    if free_exploration:
        return run_free_exploration_training(
            num_agents=num_agents,
            config=config,
            progress_bar=progress_bar,
            status_text=status_text,
            overall_status=overall_status,
            trade_metric=trade_metric,
            pnl_metric=pnl_metric,
            sentiment_csv_paths=sentiment_csv_paths,
            target_epochs=target_epochs
        )

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
        
        env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=steps_per_day, symbol=symbol_label)
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
