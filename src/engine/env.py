import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from src.engine.financial import FinancialEngine
import talib

class ForexEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Forex Trading.
    Enhanced with Technical Indicators and Dynamic Lot Sizing.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, engine: FinancialEngine, initial_balance=10000, steps_per_day=96):
        super(ForexEnv, self).__init__()
        
        self.df = self._prepare_indicators(df)
        self.engine = engine
        self.initial_balance = initial_balance
        self.steps_per_day = steps_per_day
        
        # Actions: 
        # 1. Action Type (4): 0: Hold, 1: Buy, 2: Sell, 3: Close
        # 2. Lot Size Index (3): 0: 0.01 (Micro), 1: 0.05 (Mini), 2: 0.10 (Standard-Micro)
        self.action_space = spaces.MultiDiscrete([4, 3])
        
        # Observation space: 
        # [rel_high, rel_low, rel_close, vol, pos_type, unreal_pnl, progress, sentiment, sma20_rel, sma50_rel, rsi]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.score = 0.0
        self.steps_since_last_profit = 0
        
        # Take Profit / Stop Loss (Pips)
        self.tp_pips = 50 
        self.sl_pips = 30
        
        # Ensure Sentiment column exists
        if 'sentiment' not in self.df.columns:
            self.df['sentiment'] = 0.0
        
        self.current_position = None # None, 'BUY', 'SELL'
        self.entry_price = 0.0
        self.margin_invested = 0.0
        self.lots = 0.01 # Default
        
    def _prepare_indicators(self, df):
        # Calculate Technical Indicators
        close = df['Close'].values.astype(float)
        df['sma20'] = talib.SMA(close, timeperiod=20)
        df['sma50'] = talib.SMA(close, timeperiod=50)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # Replace NaNs
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        return df

    def _next_observation(self):
        # Current row data
        current_row = self.df.iloc[self.current_step]
        
        # Relative OHLC (percentage change from Open)
        open_val = float(current_row['Open'])
        high_val = float(current_row['High'])
        low_val = float(current_row['Low'])
        close_val = float(current_row['Close'])
        
        rel_high = (high_val - open_val) / open_val
        rel_low = (low_val - open_val) / open_val
        rel_close = (close_val - open_val) / open_val
        
        # Indicators relative to price
        sma20_rel = (float(current_row['sma20']) - close_val) / close_val if close_val != 0 else 0
        sma50_rel = (float(current_row['sma50']) - close_val) / close_val if close_val != 0 else 0
        rsi = float(current_row['rsi']) / 100.0 # Normalized 0-1
        
        obs = np.array([
            rel_high,
            rel_low,
            rel_close,
            float(current_row['Volume']) / 1000.0,
            1.0 if self.current_position == 'BUY' else (-1.0 if self.current_position == 'SELL' else 0.0),
            float(self._get_unrealized_pnl()) / self.initial_balance,
            float(self.current_step) / len(self.df),
            float(current_row['sentiment']),
            sma20_rel,
            sma50_rel,
            rsi
        ], dtype=np.float32)
        return obs

    def _get_unrealized_pnl(self):
        if not self.current_position:
            return 0.0
        current_price = float(self.df.iloc[self.current_step]['Close'])
        pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price)
        fee = self.engine.calculate_brokerage_fee(self.margin_invested)
        return float(pnl - fee)

    def _check_tp_sl(self, current_price):
        if not self.current_position:
            return False
            
        # Pips estimation: 0.0001 for most pairs
        pip_value = 0.0001
        diff = current_price - self.entry_price
        
        if self.current_position == 'BUY':
            if diff >= (self.tp_pips * pip_value): return True # TP
            if diff <= -(self.sl_pips * pip_value): return True # SL
        elif self.current_position == 'SELL':
            if diff <= -(self.tp_pips * pip_value): return True # TP
            if diff >= (self.sl_pips * pip_value): return True # SL
            
        return False

    def step(self, action_vec):
        action = action_vec[0]
        lot_idx = action_vec[1]
        
        # Map lot_idx to lot values
        lot_map = {0: 0.01, 1: 0.05, 2: 0.10}
        selected_lots = lot_map.get(lot_idx, 0.01)
        
        current_price = float(self.df.iloc[self.current_step]['Close'])
        reward = 0.0
        
        # Check Automatic TP/SL first
        if self.current_position and self._check_tp_sl(current_price):
            action = 3 # Force CLOSE if TP/SL hit
        
        # Execute action
        if action == 1: # BUY
            if not self.current_position:
                self.lots = selected_lots
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'BUY'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee']
                reward -= (math_details['brokerage_fee'] / self.initial_balance) * 10
                
        elif action == 2: # SELL
            if not self.current_position:
                self.lots = selected_lots
                math_details = self.engine.open_position_math(self.lots, current_price)
                self.current_position = 'SELL'
                self.entry_price = current_price
                self.margin_invested = math_details['margin_invested']
                self.balance -= math_details['brokerage_fee']
                reward -= (math_details['brokerage_fee'] / self.initial_balance) * 10
                
        elif action == 3: # CLOSE
            if self.current_position:
                pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price)
                self.balance += pnl
                
                net_profit = pnl - self.engine.calculate_brokerage_fee(self.margin_invested)
                reward = (net_profit / self.initial_balance) * 100 
                
                profit_pct = (net_profit / self.margin_invested) * 100
                if profit_pct > 0:
                    self.score += int(profit_pct)
                    self.steps_since_last_profit = 0
                
                self.current_position = None
                self.entry_price = 0.0
                self.margin_invested = 0.0
        
        if self.current_position:
            unrealized = self._get_unrealized_pnl()
            reward += (unrealized / self.initial_balance) * 0.1
        
        self.current_step += 1
        self.steps_since_last_profit += 1
        
        if self.steps_since_last_profit >= self.steps_per_day * 5:
            self.score -= 0.1
            reward -= 0.01 
            self.steps_since_last_profit = 0
        
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0
        truncated = False
        
        obs = self._next_observation()
        info = {'balance': self.balance, 'position': self.current_position, 'score': self.score, 'lots': self.lots}
        
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
