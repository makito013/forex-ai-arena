import yfinance as yf
import pandas as pd

class MarketDataFetcher:
    def __init__(self, config):
        self.config = config
        self.pairs = [pair['symbol'] for pair in self.config['assets']['pairs']]

    def fetch_current_prices(self) -> dict:
        """
        Fetches the very latest price for all configured pairs.
        """
        prices = {}
        # Fetching all pairs simultaneously using yf.download is faster
        if not self.pairs:
            return prices
            
        data = yf.download(self.pairs, period="1d", interval="1m", group_by="ticker", progress=False)
        
        if data.empty:
            print("Warning: yfinance returned empty data for current prices.")
            return {symbol: None for symbol in self.pairs}

        for symbol in self.pairs:
            try:
                # If multiple pairs, yfinance returns a MultiIndex column DataFrame
                if len(self.pairs) > 1:
                    if symbol in data.columns.levels[0]:
                        latest_close = data[symbol]['Close'].dropna().iloc[-1]
                        prices[symbol] = float(latest_close)
                    else:
                        prices[symbol] = None
                else:
                    if 'Close' in data.columns:
                        latest_close = data['Close'].dropna().iloc[-1]
                        prices[symbol] = float(latest_close)
                    else:
                        prices[symbol] = None
            except Exception as e:
                print(f"Error extracting price for {symbol}: {e}")
                prices[symbol] = None
        
        return prices

    def fetch_historical_data(self, symbol: str, period="5d", interval="15m") -> pd.DataFrame:
        """
        Fetches historical candlestick data to feed the RL agents.
        """
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data

    def fetch_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads historical data from a local CSV file.
        Attempts to map common MetaTrader / standard column names.
        """
        encodings = ['utf-8', 'utf-16', 'latin-1']
        df = None
        
        for enc in encodings:
            try:
                # Try reading the first few lines to check for headers
                df_peek = pd.read_csv(file_path, encoding=enc, nrows=2)
                
                # Check if first row contains typical header words
                has_header = any(isinstance(col, str) and col.strip().lower() in ['date', 'time', 'open', 'high', 'low', 'close', 'vol'] 
                                for col in df_peek.columns)
                
                if has_header:
                    df = pd.read_csv(file_path, encoding=enc)
                else:
                    # MT4/MT5 typical export: Date, Time, Open, High, Low, Close, Volume
                    # Or: Datetime, Open, High, Low, Close, Volume, Spread
                    df = pd.read_csv(file_path, encoding=enc, header=None)
                    
                    if len(df.columns) >= 6:
                        # Common mapping for 6+ columns (Date, Time, O, H, L, C...) or (DateTime, O, H, L, C, V...)
                        # Let's try to detect if col 0 and 1 are Date and Time
                        try:
                            pd.to_datetime(df.iloc[0, 0] + " " + df.iloc[0, 1])
                            # It's separate Date and Time
                            df[0] = df[0] + " " + df[1]
                            df = df.drop(columns=[1])
                            df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[6:])
                        except:
                            # It's likely Datetime in col 0
                            df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[6:])
                
                if df is not None:
                    break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if df is None:
            print(f"Failed to read CSV {file_path} with any supported encoding.")
            return pd.DataFrame()

        try:
            # Normalize column names for mapping
            orig_cols = df.columns
            lower_cols = [str(c).strip().lower() for c in orig_cols]
            df.columns = lower_cols
            
            col_map = {}
            for orig, lower in zip(orig_cols, lower_cols):
                if 'date' in lower or 'time' in lower:
                    col_map[lower] = 'Datetime'
                elif 'open' in lower:
                    col_map[lower] = 'Open'
                elif 'high' in lower:
                    col_map[lower] = 'High'
                elif 'low' in lower:
                    col_map[lower] = 'Low'
                elif 'close' in lower:
                    col_map[lower] = 'Close'
                elif 'vol' in lower or 'tick' in lower:
                    col_map[lower] = 'Volume'
                    
            df = df.rename(columns=col_map)
            
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df.set_index('Datetime', inplace=True)
                
            # Fallback if Volume is missing
            if 'Volume' not in df.columns:
                df['Volume'] = 0.0
                
            # Ensure required columns exist
            req_cols = ['Open', 'High', 'Low', 'Close']
            for req in req_cols:
                if req not in df.columns:
                    raise ValueError(f"Missing required column '{req}' in CSV. Found: {list(df.columns)}")
            
            # Ensure data types are float
            for col in req_cols + ['Volume']:
                df[col] = df[col].astype(float)
                
            # Drop rows with NaNs in critical columns
            df.dropna(subset=req_cols, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
            return pd.DataFrame()

    def load_sentiment_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Expects a CSV with columns like: Datetime, Score (or Sentiment)
        OR handles the Kaggle 'Calender_data.csv' format.
        """
        # Read the first few lines to detect Kaggle format
        try:
            df_peek = pd.read_csv(file_path, nrows=5)
            is_kaggle_format = 'importance' in df_peek.columns and 'event' in df_peek.columns
        except:
            is_kaggle_format = False

        if is_kaggle_format:
            # Special handling for Kaggle Economic Calendar
            # We'll use a larger encoding list just in case
            df = None
            for enc in ['utf-8', 'latin-1', 'utf-16']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except: continue
            
            if df is None or df.empty: return pd.DataFrame()

            # Create Datetime from 'date' and 'time'
            # Filter out 'All Day' or handle them as 00:00
            df['time'] = df['time'].replace('All Day', '00:00')
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce', dayfirst=True)
            df = df.dropna(subset=['Datetime'])
            df.set_index('Datetime', inplace=True)

            # Map importance to a simple sentiment/volatility score
            # high = 0.5, medium = 0.2, low = 0.1 (absolute impact)
            # This is a simplification; a real LLM would judge if the news is good or bad.
            # But for training, "importance" is a good proxy for expected volatility.
            importance_map = {'high': 0.5, 'medium': 0.2, 'low': 0.1}
            df['sentiment'] = df['importance'].str.lower().map(importance_map).fillna(0.0)
            
            # Group by Datetime to handle multiple events at the same time (take max importance)
            df = df.groupby(level=0)[['sentiment']].max()
            return df

        # Default standard logic
        df = self.fetch_from_csv(file_path)
        if df.empty:
            return df
            
        # Common mapping for sentiment column
        col_map = {}
        for col in df.columns:
            if 'score' in col or 'sentiment' in col or 'value' in col:
                col_map[col] = 'sentiment'
                
        df = df.rename(columns=col_map)
        
        if 'sentiment' not in df.columns and len(df.columns) > 0:
            # Fallback to the first numeric column that isn't OHLCV
            potential_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            if potential_cols:
                df = df.rename(columns={potential_cols[0]: 'sentiment'})
        
        return df[['sentiment']] if 'sentiment' in df.columns else pd.DataFrame()

    def attach_sentiment_to_df(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Joins sentiment data into price data. 
        Uses 'forward fill' so a sentiment score stays valid until the next one appears.
        """
        if sentiment_df.empty:
            price_df['sentiment'] = 0.0
            return price_df
            
        # Merge on Datetime index
        combined = price_df.merge(sentiment_df, left_index=True, right_index=True, how='left')
        
        # Fill missing values: first forward fill, then fill remaining (at start) with 0.0
        combined['sentiment'] = combined['sentiment'].ffill().fillna(0.0)
        
        return combined
