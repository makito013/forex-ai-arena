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
        
        # Actions: [ActionType(4), LotSizeIndex(3)]
        self.action_space = spaces.MultiDiscrete([4, 3])
        
        # Expanded Observation Space (26 variables):
        # 1-3: Rel OHLC, 4: Vol, 5: Pos, 6: PnL, 7: Progress, 8: Sentiment
        # 9-11: EMA Rel (21, 50, 200)
        # 12-14: MACD (Line, Signal, Hist)
        # 15: RSI, 16-17: Stoch (K, D)
        # 18-19: Bollinger (Upper/Lower Rel)
        # 20-21: Ichimoku (Kumo Top/Bottom Rel)
        # 22: Pin Bar Signal (0/1)
        # 23-26: MTF (ema50_h4, rsi_h4, ema50_d1, rsi_d1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.score = 0.0
        self.steps_since_last_profit = 0
        
        self.tp_pips = 50 
        self.sl_pips = 30
        
        if 'sentiment' not in self.df.columns:
            self.df['sentiment'] = 0.0
        
        self.current_position = None
        self.entry_price = 0.0
        self.margin_invested = 0.0
        self.lots = 0.01

    def _prepare_indicators(self, df):
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        
        # Strategy 1, 2, 10: EMAs
        df['ema21'] = talib.EMA(close, timeperiod=21)
        df['ema50'] = talib.EMA(close, timeperiod=50)
        df['ema200'] = talib.EMA(close, timeperiod=200)
        
        # Strategy 2: MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Strategy 4: RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # Strategy 5: Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Strategy 6: Bollinger Bands
        df['bb_up'], df['bb_mid'], df['bb_low'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Strategy 3: Ichimoku (simplified for current observation)
        # Tenkan: 9, Kijun: 26, Senkou B: 52
        df['tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        # Senkou Span A (plotted 26 ahead, so we look at value from 26 periods ago)
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        # Senkou Span B (plotted 26 ahead)
        df['span_b'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        
        # Strategy 9: Pin Bar Detection
        body = np.abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        df['pin_bar'] = ((body < total_range * 0.3) & ((df['High'] - np.maximum(df['Open'], df['Close'])) > total_range * 0.5) | 
                         ((np.minimum(df['Open'], df['Close']) - df['Low']) > total_range * 0.5)).astype(float)
        
        # Strategy 7: Breakout (Rolling High/Low)
        df['rolling_high_50'] = df['High'].rolling(50).max()
        df['rolling_low_50'] = df['Low'].rolling(50).min()
        
        # --- Multi-Timeframe (MTF) Automation ---
        # Resample to H4 and D1 if we have a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            # Resample to H4 (4 hours)
            df_h4 = df.resample('4H').agg({'Close': 'last'}).dropna()
            if not df_h4.empty and len(df_h4) > 50:
                df_h4['ema50_h4'] = talib.EMA(df_h4['Close'].values, timeperiod=50)
                df_h4['rsi_h4'] = talib.RSI(df_h4['Close'].values, timeperiod=14)
                df = df.join(df_h4[['ema50_h4', 'rsi_h4']], how='left')
                df['ema50_h4'] = df['ema50_h4'].ffill()
                df['rsi_h4'] = df['rsi_h4'].ffill()
            else:
                df['ema50_h4'] = df['ema50']
                df['rsi_h4'] = df['rsi']
                
            # Resample to D1 (1 Day)
            df_d1 = df.resample('1D').agg({'Close': 'last'}).dropna()
            if not df_d1.empty and len(df_d1) > 50:
                df_d1['ema50_d1'] = talib.EMA(df_d1['Close'].values, timeperiod=50)
                df_d1['rsi_d1'] = talib.RSI(df_d1['Close'].values, timeperiod=14)
                df = df.join(df_d1[['ema50_d1', 'rsi_d1']], how='left')
                df['ema50_d1'] = df['ema50_d1'].ffill()
                df['rsi_d1'] = df['rsi_d1'].ffill()
            else:
                df['ema50_d1'] = df['ema50']
                df['rsi_d1'] = df['rsi']
        else:
            df['ema50_h4'] = df['ema50']
            df['rsi_h4'] = df['rsi']
            df['ema50_d1'] = df['ema50']
            df['rsi_d1'] = df['rsi']
        
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        return df

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        open_val = float(row['Open'])
        close_val = float(row['Close'])
        
        # Helper for relative values
        rel = lambda val: (float(val) - close_val) / close_val if close_val != 0 else 0
        
        obs = [
            (float(row['High']) - open_val) / open_val,
            (float(row['Low']) - open_val) / open_val,
            (close_val - open_val) / open_val,
            float(row['Volume']) / 1000.0,
            1.0 if self.current_position == 'BUY' else (-1.0 if self.current_position == 'SELL' else 0.0),
            float(self._get_unrealized_pnl()) / self.initial_balance,
            float(self.current_step) / len(self.df),
            float(row['sentiment']),
            rel(row['ema21']),
            rel(row['ema50']),
            rel(row['ema200']),
            float(row['macd_hist']) / close_val if close_val != 0 else 0,
            float(row['rsi']) / 100.0,
            float(row['stoch_k']) / 100.0,
            float(row['stoch_d']) / 100.0,
            rel(row['bb_up']),
            rel(row['bb_low']),
            rel(np.maximum(row['span_a'], row['span_b'])), # Kumo Top
            rel(np.minimum(row['span_a'], row['span_b'])), # Kumo Bottom
            rel(row['rolling_high_50']), # Dist to 50-bar High
            rel(row['rolling_low_50']),  # Dist to 50-bar Low
            float(row['pin_bar']),
            rel(row['ema50_h4']),        # MTF: H4 EMA50
            float(row['rsi_h4']) / 100.0,# MTF: H4 RSI
            rel(row['ema50_d1']),        # MTF: D1 EMA50
            float(row['rsi_d1']) / 100.0 # MTF: D1 RSI
        ]
        return np.array(obs, dtype=np.float32)

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
