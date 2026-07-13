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

    def __init__(self, df: pd.DataFrame, engine: FinancialEngine, initial_balance=10000, steps_per_day=96, symbol=None):
        super(ForexEnv, self).__init__()

        self.df = self._prepare_indicators(df)
        self.engine = engine
        self.initial_balance = initial_balance
        self.steps_per_day = steps_per_day
        self.symbol = symbol
        # Units per lot for this asset (Gold = 100 oz/lot vs Forex 100,000 units/lot)
        self.contract_size = engine.get_contract_size(symbol)
        # USD-base pairs (USDCHF, USDCAD, USDJPY) need different margin/PnL conversion
        self.is_usd_base = engine.is_usd_base(symbol)
        # Realistic costs (spread+commission+swap) vs legacy 5%-of-margin fee
        self.costs_enabled = engine.costs_enabled()

        # Risk Management rules (real trading constraints, visible to the agent)
        trading_cfg = engine.config.get('trading', {})
        self.max_margin_usage = float(trading_cfg.get('max_margin_usage_pct', 0.5))
        self.stop_out_level = float(trading_cfg.get('stop_out_level', 0.5))
        self.bankruptcy_threshold = 0.05  # Episode ends if equity drops below 5% of initial balance

        # Risk-aware reward shaping: make drawdown actually hurt the reward & score
        self.dd_limit = float(trading_cfg.get('max_drawdown_limit', 0.30))
        self.dd_penalty_scale = float(trading_cfg.get('dd_penalty_scale', 30.0))
        self.dd_breach_scale = float(trading_cfg.get('dd_breach_scale', 10.0))

        # Actions: [ActionType(4), LotSizeIndex(3)]
        self.action_space = spaces.MultiDiscrete([4, 3])

        # Expanded Observation Space (29 variables):
        # 1-3: Rel OHLC, 4: Vol, 5: Pos, 6: PnL, 7: Progress, 8: Sentiment
        # 9-11: EMA Rel (21, 50, 200)
        # 12-14: MACD (Line, Signal, Hist)
        # 15: RSI, 16-17: Stoch (K, D)
        # 18-19: Bollinger (Upper/Lower Rel)
        # 20-21: Ichimoku (Kumo Top/Bottom Rel)
        # 22: Pin Bar Signal (0/1)
        # 23-26: MTF (ema50_h4, rsi_h4, ema50_d1, rsi_d1)
        # 27: Equity ratio (equity / initial balance)
        # 28: Margin usage (margin invested / equity)
        # 29: Current drawdown from equity peak
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        self.current_step = 0
        self.balance = self.initial_balance
        self.score = 0.0
        self.steps_since_last_profit = 0

        # Drawdown tracking (max_drawdown persists across resets: worst ever for this env)
        self.equity_peak = self.initial_balance
        self.max_drawdown = 0.0
        self.episode_max_dd = 0.0  # per-episode worst DD, used as the reward-penalty reference
        
        self.tp_pips = 50 
        self.sl_pips = 30
        
        if 'sentiment' not in self.df.columns:
            self.df['sentiment'] = 0.0
        
        self.current_position = None
        self.entry_price = 0.0
        self.margin_invested = 0.0
        self.lots = 0.01
        self.open_cost = 0.0   # spread+commission paid when the current position was opened
        self.entry_step = 0    # step index at which the position was opened (for swap)

    @staticmethod
    def _prepare_indicators(df):
        # Skip recomputation if this DataFrame was already prepared (cached across epochs)
        if 'rsi_d1' in df.columns:
            return df
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

        # Risk Management observations: the agent SEES its equity, margin usage and drawdown
        equity = self._get_equity()
        margin_usage = (self.margin_invested / equity) if (self.current_position and equity > 0) else 0.0
        current_dd = (self.equity_peak - equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        obs.extend([
            float(equity) / self.initial_balance,        # Equity ratio
            float(margin_usage),                         # Margin usage
            float(min(1.0, max(0.0, current_dd)))        # Current drawdown from peak (0..1)
        ])
        return np.array(obs, dtype=np.float32)

    def _get_unrealized_pnl(self):
        if not self.current_position:
            return 0.0
        current_price = float(self.df.iloc[self.current_step]['Close'])
        pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price, contract_size=self.contract_size, is_base_usd=self.is_usd_base)
        if self.costs_enabled:
            nights = int((self.current_step - self.entry_step) / max(self.steps_per_day, 1))
            swap = self.engine.swap_cost(self.lots, nights)
            return float(pnl - self.open_cost - swap)  # spread+commission already paid at open
        fee = self.engine.calculate_brokerage_fee(self.margin_invested)
        return float(pnl - fee)

    def _get_equity(self):
        """Equity = balance + unrealized PnL of the open position."""
        return self.balance + self._get_unrealized_pnl()

    def _can_open(self, math_details, open_cost=None):
        """
        Margin rules (like a real broker):
        1. Balance must cover the required margin PLUS the opening cost (spread+commission, or legacy fee).
        2. The required margin can't exceed max_margin_usage of equity (no all-in).
        """
        margin = math_details['margin_invested']
        cost = open_cost if open_cost is not None else math_details['brokerage_fee']
        equity = self._get_equity()
        if margin + cost > self.balance:
            return False
        if margin > equity * self.max_margin_usage:
            return False
        return True

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

        # Risk Management: Broker Stop-Out (Margin Call) — checked BEFORE the agent acts
        stopped_out = False
        if self.current_position and self.margin_invested > 0:
            margin_level = self._get_equity() / self.margin_invested
            if margin_level <= self.stop_out_level:
                action = 3  # Broker force-closes the position
                stopped_out = True

        # Check Automatic TP/SL first
        if not stopped_out and self.current_position and self._check_tp_sl(current_price):
            action = 3 # Force CLOSE if TP/SL hit

        # Execute action
        if action in (1, 2): # BUY / SELL
            if not self.current_position:
                math_details = self.engine.open_position_math(selected_lots, current_price, contract_size=self.contract_size, is_base_usd=self.is_usd_base)
                # Real opening cost: spread+commission (or legacy 5%-of-margin fee)
                if self.costs_enabled:
                    open_cost = self.engine.transaction_cost(selected_lots, current_price, symbol=self.symbol, contract_size=self.contract_size, is_base_usd=self.is_usd_base)
                else:
                    open_cost = math_details['brokerage_fee']

                if self._can_open(math_details, open_cost):
                    self.lots = selected_lots
                    self.current_position = 'BUY' if action == 1 else 'SELL'
                    self.entry_price = current_price
                    self.margin_invested = math_details['margin_invested']
                    self.open_cost = open_cost
                    self.entry_step = self.current_step
                    self.balance -= open_cost
                    reward -= (open_cost / self.initial_balance) * 10
                else:
                    reward -= 0.05  # Order rejected: insufficient free margin (teaches margin discipline)

        elif action == 3: # CLOSE
            if self.current_position:
                pnl = self.engine.calculate_pnl(self.current_position, self.lots, self.entry_price, current_price, contract_size=self.contract_size, is_base_usd=self.is_usd_base)
                self.balance += pnl

                # Overnight swap for the nights the position was held (real cost model only)
                if self.costs_enabled:
                    nights = int((self.current_step - self.entry_step) / max(self.steps_per_day, 1))
                    swap = self.engine.swap_cost(self.lots, nights)
                    self.balance -= swap
                    net_profit = pnl - self.open_cost - swap  # open_cost was already deducted at open
                else:
                    net_profit = pnl - self.engine.calculate_brokerage_fee(self.margin_invested)

                # Negative Balance Protection (standard retail broker rule):
                # a gap can make the loss exceed the balance; the account floors at 0.
                if self.balance < 0:
                    self.balance = 0.0

                reward = (net_profit / self.initial_balance) * 100

                profit_pct = (net_profit / self.margin_invested) * 100 if self.margin_invested > 0 else 0.0
                if profit_pct > 0:
                    self.score += int(profit_pct)
                    self.steps_since_last_profit = 0
                
                self.current_position = None
                self.entry_price = 0.0
                self.margin_invested = 0.0
                self.open_cost = 0.0

        if stopped_out:
            reward -= 1.0  # Extra penalty: the broker had to liquidate (margin call)

        if self.current_position:
            unrealized = self._get_unrealized_pnl()
            reward += (unrealized / self.initial_balance) * 0.1

        # Track Equity Peak & Max Drawdown, and SHAPE THE REWARD to discourage drawdown
        equity = self._get_equity()
        if equity > self.equity_peak:
            self.equity_peak = equity
        current_dd = min(1.0, (self.equity_peak - equity) / self.equity_peak) if self.equity_peak > 0 else 0.0

        # 1. Penalize each NEW depth of drawdown reached this episode.
        #    Telescopes to roughly (episode_max_dd * scale) of total penalty over the episode.
        if current_dd > self.episode_max_dd:
            reward -= (current_dd - self.episode_max_dd) * self.dd_penalty_scale
            self.episode_max_dd = current_dd

        # 2. Strong, continuous penalty for every step spent BEYOND the allowed limit
        #    (this is the "score negative when drawdown passes the max" idea).
        if current_dd > self.dd_limit:
            over = current_dd - self.dd_limit
            reward -= over * self.dd_breach_scale
            self.score -= over  # also drags down the evolutionary score

        # Reporting: worst drawdown ever seen by this env (persists across resets)
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        self.current_step += 1
        self.steps_since_last_profit += 1

        if self.steps_since_last_profit >= self.steps_per_day * 5:
            self.score -= 0.1
            reward -= 0.01
            self.steps_since_last_profit = 0

        # Bankruptcy: equity wiped out (below 5% of initial) -> the agent can no longer trade
        bankrupt = equity <= self.initial_balance * self.bankruptcy_threshold
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0 or bankrupt
        truncated = False

        obs = self._next_observation()
        info = {'balance': self.balance, 'position': self.current_position, 'score': self.score, 'lots': self.lots,
                'equity': equity, 'max_drawdown': self.max_drawdown, 'stopped_out': stopped_out, 'bankrupt': bankrupt}

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
        self.open_cost = 0.0
        self.entry_step = 0
        # equity_peak & episode_max_dd reset per episode; max_drawdown persists (worst ever)
        self.equity_peak = self.initial_balance
        self.episode_max_dd = 0.0
        return self._next_observation(), {}
