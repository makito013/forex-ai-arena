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
        
        for symbol in self.pairs:
            try:
                # If multiple pairs, yfinance returns a MultiIndex column DataFrame
                if len(self.pairs) > 1:
                    latest_close = data[symbol]['Close'].iloc[-1]
                else:
                    latest_close = data['Close'].iloc[-1]
                prices[symbol] = float(latest_close)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
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
        try:
            df = pd.read_csv(file_path)
            
            # Normalize column names for mapping
            orig_cols = df.columns
            lower_cols = [c.strip().lower() for c in orig_cols]
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
