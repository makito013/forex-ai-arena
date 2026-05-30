import os
import sys
import yaml
import string
import random
from stable_baselines3 import PPO

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetcher import MarketDataFetcher
from src.engine.financial import FinancialEngine
from src.engine.env import ForexEnv
from src.db_models import init_db, Agent

def generate_agent_name():
    return "Agent_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

def train():
    print("Loading Configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    engine = FinancialEngine('config.yaml')
    fetcher = MarketDataFetcher(config)
    session = init_db()

    target_symbol = "EURUSD=X" # Train on EUR/USD for now
    print(f"Fetching historical data for {target_symbol}...")
    # Get 5 days of 15-minute intervals
    df = fetcher.fetch_historical_data(target_symbol, period="5d", interval="1m")
    
    if df.empty:
        print("Failed to fetch data. Cannot train.")
        return

    print(f"Initializing Environment with {len(df)} steps...")
    env = ForexEnv(df=df, engine=engine, initial_balance=config['arena']['initial_balance'])
    
    agent_name = generate_agent_name()
    print(f"Training PPO Agent: {agent_name}...")
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train for a small number of timesteps for demonstration
    # In a real scenario, this would be 100,000 to 1,000,000+
    timesteps = min(10000, len(df) * 2) 
    model.learn(total_timesteps=timesteps)
    
    print("Training Complete. Saving to Database...")
    # Evaluate final balance (simplistic approach: just look at the env's final state)
    final_balance = env.balance
    
    new_agent = Agent(name=agent_name, strategy_type="RL_PPO", balance=final_balance)
    session.add(new_agent)
    session.commit()
    
    # Save the model weights
    os.makedirs('models', exist_ok=True)
    model.save(f"models/{agent_name}")
    print(f"Model saved to models/{agent_name}.zip")
    print(f"Agent logged to Database with Final Balance: ${final_balance:.2f}")

if __name__ == "__main__":
    train()
