import os
import string
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.data.fetcher import MarketDataFetcher
from src.engine.financial import FinancialEngine
from src.engine.env import ForexEnv
from src.db_models import init_db, Agent

class StreamlitCallback(BaseCallback):
    """
    Custom callback for reporting training progress to the Streamlit UI.
    """
    def __init__(self, progress_bar, status_text, total_timesteps, agent_name, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        # Streamlit progress bars must be between 0.0 and 1.0
        safe_progress = max(0.0, min(progress, 1.0))
        self.progress_bar.progress(safe_progress)
        self.status_text.text(f"Training {self.agent_name} | Step: {self.num_timesteps}/{self.total_timesteps}")
        return True

def generate_agent_name():
    return "Agent_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

def run_training_session(symbol, period, interval, num_agents, config, progress_bar, status_text, overall_status, existing_agent_name=None):
    engine = FinancialEngine('config.yaml')
    fetcher = MarketDataFetcher(config)
    session = init_db()
    
    overall_status.info(f"Fetching historical data for {symbol} ({period} / {interval})...")
    df = fetcher.fetch_historical_data(symbol, period=period, interval=interval)
    
    if df.empty:
        overall_status.error(f"Failed to fetch data for {symbol}. Training aborted.")
        return False, []

    data_length = len(df)
    
    if existing_agent_name:
        num_agents = 1
        overall_status.success(f"Data fetched! {data_length} candles. Continuing training for {existing_agent_name}.")
    else:
        overall_status.success(f"Data fetched! {data_length} candles available. Training {num_agents} new agent(s).")
    
    results = []
    
    for i in range(num_agents):
        if existing_agent_name:
            agent_name = existing_agent_name
            db_agent = session.query(Agent).filter_by(name=agent_name).first()
            initial_balance = db_agent.balance if db_agent else config['arena']['initial_balance']
        else:
            agent_name = generate_agent_name()
            initial_balance = config['arena']['initial_balance']
            
        overall_status.info(f"Initializing Environment for Agent {i+1}/{num_agents}: {agent_name}...")
        
        env = ForexEnv(df=df, engine=engine, initial_balance=initial_balance)
        
        model_path = f"models/{agent_name}.zip"
        if existing_agent_name and os.path.exists(model_path):
            overall_status.info(f"Loading existing model weights for {agent_name}...")
            model = PPO.load(model_path, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=0)
        
        # Train for approximately 3 epochs over the dataset
        timesteps = min(50000, data_length * 3)
        
        # Setup callback to link to Streamlit UI
        callback = StreamlitCallback(progress_bar, status_text, timesteps, agent_name)
        
        model.learn(total_timesteps=timesteps, callback=callback)
        
        final_balance = env.balance
        strategy_label = f"PPO_{symbol}_{period}"
        
        if existing_agent_name and db_agent:
            db_agent.balance = final_balance
            if symbol not in db_agent.strategy_type:
                db_agent.strategy_type += f" + {symbol}"
            strategy_label = db_agent.strategy_type
        else:
            new_agent = Agent(name=agent_name, strategy_type=strategy_label, balance=final_balance)
            session.add(new_agent)
            
        session.commit()
        
        os.makedirs('models', exist_ok=True)
        model.save(f"models/{agent_name}")
        
        results.append({
            "Agent": agent_name,
            "Symbol": symbol,
            "Strategy": strategy_label,
            "Final Balance": round(final_balance, 2)
        })
        
    return True, results
