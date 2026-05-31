import os
from stable_baselines3 import PPO

from src.data.fetcher import MarketDataFetcher
from src.engine.financial import FinancialEngine
from src.engine.env import ForexEnv
from src.db_models import init_db, Agent

def get_steps_per_day(interval):
    mapping = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "1d": 1
    }
    return mapping.get(interval, 96)

def run_competition(agent_names, symbol, period, interval, config, progress_bar, status_text, use_csv=False, csv_path=None):
    engine = FinancialEngine('config.yaml')
    fetcher = MarketDataFetcher(config)
    session = init_db()
    
    if use_csv and csv_path:
        status_text.text(f"Loading competition data from CSV: {csv_path}...")
        df = fetcher.fetch_from_csv(csv_path)
    else:
        status_text.text(f"Fetching competition data for {symbol}...")
        df = fetcher.fetch_historical_data(symbol, period=period, interval=interval)
    
    if df.empty:
        status_text.text("Failed to load competition data.")
        return False, []

    steps_per_day = get_steps_per_day(interval)
    
    results = []
    total_agents = len(agent_names)
    
    for i, agent_name in enumerate(agent_names):
        progress_bar.progress(i / total_agents)
        status_text.text(f"Running Agent {agent_name} in Competition Arena...")
        
        db_agent = session.query(Agent).filter_by(name=agent_name).first()
        if not db_agent:
            continue
            
        model_path = f"models/{agent_name}.zip"
        if not os.path.exists(model_path):
            continue
            
        env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'], steps_per_day=steps_per_day)
        
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Error loading {agent_name}: {e}")
            continue
            
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        # Update global score in DB based on competition results
        db_agent.score += env.score
        session.commit()
        
        results.append({
            "Agent": agent_name,
            "Strategy": db_agent.strategy_type,
            "Final Balance ($)": round(env.balance, 2),
            "Competition Score": int(env.score)
        })
        
    progress_bar.progress(1.0)
    status_text.text("Competition Finished!")
    
    # Sort by competition score (descending)
    results = sorted(results, key=lambda x: x["Competition Score"], reverse=True)
    return True, results
