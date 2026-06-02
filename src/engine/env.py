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

    def __init__(self, df: pd.DataFrame, engine: FinancialEngine, initial_balance=10000, steps_per_day=96):
        super(ForexEnv, self).__init__()
        
        self.df = df
        self.engine = engine
        self.initial_balance = initial_balance
        self.steps_per_day = steps_per_day
        
        # Actions: 0: Hold, 1: Buy (Long), 2: Sell (Short), 3: Close Position
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 
        # [rel_high, rel_low, rel_close, vol, pos_type, unrealized_pnl, progress, sentiment]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.score = 0.0
        self.steps_since_last_profit = 0
        
        # Ensure Sentiment column exists
        if 'sentiment' not in self.df.columns:
            self.df['sentiment'] = 0.0
        
        self.current_position = None # None, 'BUY', 'SELL'
        self.entry_price = 0.0
        self.margin_invested = 0.0
        self.lots = self.engine.config['trading']['default_micro_lots']
        
    def _next_observation(self):
        # Current row data
        current_row = self.df.iloc[self.current_step]
        
        # Relative OHLC (percentage change from Open)
        open_val = float(np.squeeze(current_row['Open']))
        high_val = float(np.squeeze(current_row['High']))
        low_val = float(np.squeeze(current_row['Low']))
        close_val = float(np.squeeze(current_row['Close']))
        
        rel_high = (high_val - open_val) / open_val
        rel_low = (low_val - open_val) / open_val
        rel_close = (close_val - open_val) / open_val
        
        obs = np.array([
            rel_high,
            rel_low,
            rel_close,
            float(np.squeeze(current_row['Volume'])) / 1000.0, # Scaled volume
            1.0 if self.current_position == 'BUY' else (-1.0 if self.current_position == 'SELL' else 0.0),
            float(self._get_unrealized_pnl()) / self.initial_balance, # Scaled PnL
            float(self.current_step) / len(self.df), # Progress in data
            float(np.squeeze(current_row['sentiment'])) # Sentiment score [-1, 1]
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
        reward = 0.0
        
        # Execute action
        if action == 1: # BUY
            if not self.current_position:
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'BUY'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee'] # Pay fee immediately
                reward -= (math_details['brokerage_fee'] / self.initial_balance) * 10 # Small normalized penalty
                
        elif action == 2: # SELL
            if not self.current_position:
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'SELL'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee']
                reward -= (math_details['brokerage_fee'] / self.initial_balance) * 10
                
        elif action == 3: # CLOSE
            if self.current_position:
                pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price)
                self.balance += pnl # Fee was already deducted at opening
                
                # Reward based on Net Profit (PnL - fee)
                net_profit = pnl - self.engine.calculate_brokerage_fee(self.margin_invested)
                reward = (net_profit / self.initial_balance) * 100 # Incentivize positive net profit
                
                # Scoring Logic
                profit_pct = (net_profit / self.margin_invested) * 100
                if profit_pct > 0:
                    points_earned = int(profit_pct) # 1 point per 1% profit
                    self.score += points_earned
                    if points_earned > 0:
                        self.steps_since_last_profit = 0
                
                # Reset position
                self.current_position = None
                self.entry_price = 0.0
                self.margin_invested = 0.0
        
        # Hold (Action 0): Small step reward/penalty to encourage growth
        if self.current_position:
            # Unrealized PnL change as intermediate reward
            # This helps the agent understand if the current trend is good
            unrealized = self._get_unrealized_pnl()
            reward += (unrealized / self.initial_balance) * 0.1
        
        self.current_step += 1
        self.steps_since_last_profit += 1
        
        # Penalty for prolonged inactivity (reduced)
        if self.steps_since_last_profit >= self.steps_per_day * 5: # 5 days of silence
            self.score -= 0.1
            reward -= 0.01 
            self.steps_since_last_profit = 0
        
        # Check if episode is done (reached end of data or bankrupt)
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0
        truncated = False
        
        # Final observation
        obs = self._next_observation()
        
        info = {
            'balance': self.balance,
            'position': self.current_position,
            'score': self.score
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.score = 0.0
        self.steps_since_last_profit = 0
        self.current_position = None
        self.entry_price = 0.0
        self.margin_invested = 0.0
        return self._next_observation(), {}
