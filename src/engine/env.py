import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from src.engine.financial import FinancialEngine

class ForexEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Forex Trading.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, engine: FinancialEngine, initial_balance=10000):
        super(ForexEnv, self).__init__()
        
        self.df = df
        self.engine = engine
        self.initial_balance = initial_balance
        
        # Actions: 0: Hold, 1: Buy (Long), 2: Sell (Short), 3: Close Position
        self.action_space = spaces.Discrete(4)
        
        # Observation space: OHLCV + Sentiment (Placeholder for now) + Current Position
        # Let's say we look at current step's [Open, High, Low, Close, Volume, Position_Type, Unrealized_PnL]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_position = None # None, 'BUY', 'SELL'
        self.entry_price = 0.0
        self.margin_invested = 0.0
        self.lots = self.engine.config['trading']['default_micro_lots']
        
    def _next_observation(self):
        # Current row data
        obs = np.array([
            float(np.squeeze(self.df.iloc[self.current_step]['Open'])),
            float(np.squeeze(self.df.iloc[self.current_step]['High'])),
            float(np.squeeze(self.df.iloc[self.current_step]['Low'])),
            float(np.squeeze(self.df.iloc[self.current_step]['Close'])),
            float(np.squeeze(self.df.iloc[self.current_step]['Volume'])),
            1.0 if self.current_position == 'BUY' else (-1.0 if self.current_position == 'SELL' else 0.0),
            float(self._get_unrealized_pnl())
        ], dtype=np.float32)
        return obs

    def _get_unrealized_pnl(self):
        if not self.current_position:
            return 0.0
        current_price = float(np.squeeze(self.df.iloc[self.current_step]['Close']))
        pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price)
        # Deduct the initial fee to get Net Unrealized
        fee = self.engine.calculate_brokerage_fee(self.margin_invested)
        return float(pnl - fee)

    def step(self, action):
        current_price = float(np.squeeze(self.df.iloc[self.current_step]['Close']))
        reward = 0
        
        # Execute action
        if action == 1: # BUY
            if not self.current_position:
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'BUY'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee'] # Pay fee immediately
                reward = -math_details['brokerage_fee'] # Immediate small penalty for opening
                
        elif action == 2: # SELL
            if not self.current_position:
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'SELL'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee']
                reward = -math_details['brokerage_fee']
                
        elif action == 3: # CLOSE
            if self.current_position:
                pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price)
                self.balance += pnl # Fee was already deducted at opening
                reward = pnl
                
                # Reset position
                self.current_position = None
                self.entry_price = 0.0
                self.margin_invested = 0.0
        
        # Hold (Action 0) does nothing but lets time pass
        
        self.current_step += 1
        
        # Check if episode is done (reached end of data or bankrupt)
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0
        truncated = False
        
        # Final observation
        obs = self._next_observation()
        
        info = {
            'balance': self.balance,
            'position': self.current_position
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_position = None
        self.entry_price = 0.0
        self.margin_invested = 0.0
        return self._next_observation(), {}
