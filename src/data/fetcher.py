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
