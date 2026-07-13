import yaml

class FinancialEngine:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.leverage = self.config['trading']['leverage']
        self.fee_pct = self.config['trading']['brokerage_fee_pct']
        self.standard_lot = self.config['trading']['standard_lot_size']

    def apply_config(self, config: dict):
        """
        Replaces the loaded config (e.g. with a UI-overridden copy) and refreshes
        the derived attributes — assigning .config alone would NOT update leverage/fees.
        """
        self.config = config
        self.leverage = config['trading']['leverage']
        self.fee_pct = config['trading']['brokerage_fee_pct']
        self.standard_lot = config['trading']['standard_lot_size']

    def get_contract_size(self, symbol=None) -> float:
        """
        Units per standard lot for a given asset. Forex pairs default to standard_lot
        (100,000); non-forex assets (e.g. Gold = 100 oz) are overridden in config.yaml
        under trading.contract_sizes.
        """
        sizes = self.config['trading'].get('contract_sizes', {}) or {}
        if symbol:
            symbol_up = str(symbol).upper()
            for key, size in sizes.items():
                if str(key).upper() in symbol_up:
                    return float(size)
        return float(self.standard_lot)

    def is_usd_base(self, symbol=None) -> bool:
        """
        True when USD is the BASE currency of the pair (USDJPY, USDCHF, USDCAD).
        Yahoo shorthand like 'JPY=X' / 'CHF=X' also means USD/XXX.
        XXXUSD pairs (EURUSD, GBPUSD, XAUUSD) return False: USD is the quote currency.
        """
        if not symbol:
            return False
        s = str(symbol).upper()
        if s.endswith('=X') and len(s[:-2]) == 3:  # 'JPY=X' == USD/JPY
            return True
        return s.startswith('USD')

    def calculate_margin(self, lots: float, current_price: float, is_base_usd: bool = False, contract_size: float = None) -> float:
        """
        Margin (in USD) = position notional in USD / leverage.
        - USD is the base currency (USD/JPY): the units ARE dollars -> units / leverage.
        - USD is the quote currency (EUR/USD): notional = units * price -> units * price / leverage.
        """
        units = lots * (contract_size or self.standard_lot)
        if is_base_usd:
            return units / self.leverage
        return (units * current_price) / self.leverage

    def calculate_brokerage_fee(self, margin: float) -> float:
        """
        Legacy fictional fee: a percentage of the margin invested.
        Kept for backward compatibility / the Math Test tab. Real runs use the
        spread+commission+swap model below when costs.enabled is true.
        """
        return margin * self.fee_pct

    # ----- Realistic broker costs (spread / commission / swap) -----
    def costs_enabled(self) -> bool:
        return bool(self.config.get('costs', {}).get('enabled', False))

    def get_pip_size(self, symbol=None) -> float:
        """Price value of 1 pip/point: 0.01 for JPY pairs and Gold, 0.0001 otherwise."""
        s = str(symbol).upper() if symbol else ''
        if 'JPY' in s:
            return 0.01
        if 'XAU' in s or 'GC=' in s or 'GOLD' in s:
            return 0.01
        return 0.0001

    def get_spread_pips(self, symbol=None) -> float:
        costs = self.config.get('costs', {})
        overrides = costs.get('spreads_pips', {}) or {}
        if symbol:
            su = str(symbol).upper()
            for key, val in overrides.items():
                if str(key).upper() in su:
                    return float(val)
        return float(costs.get('default_spread_pips', 1.0))

    def transaction_cost(self, lots, price, symbol=None, contract_size=None, is_base_usd=False) -> float:
        """
        Round-trip transaction cost in USD = spread cost + commission.
        Spread: you buy at ask and sell at bid, so a round trip pays the full spread.
        For USD-base pairs the spread cost is in the quote currency -> converted to USD.
        """
        units = lots * (contract_size or self.standard_lot)
        spread_price = self.get_spread_pips(symbol) * self.get_pip_size(symbol)
        spread_cost = spread_price * units
        if is_base_usd and price != 0:
            spread_cost = spread_cost / price  # quote currency -> USD
        commission = float(self.config.get('costs', {}).get('commission_per_lot', 0.0)) * lots
        return spread_cost + commission

    def swap_cost(self, lots, nights) -> float:
        """
        Overnight financing for `nights` held. Returns a POSITIVE number to subtract
        (config value negative = cost, positive = credit).
        """
        rate = float(self.config.get('costs', {}).get('swap_per_lot_per_night', 0.0))
        return -rate * lots * max(0, int(nights))

    def open_position_math(self, lots: float, open_price: float, contract_size: float = None, is_base_usd: bool = False):
        """
        Returns a dictionary with margin and fee details when opening a position.
        """
        margin = self.calculate_margin(lots, open_price, is_base_usd=is_base_usd, contract_size=contract_size)
        fee = self.calculate_brokerage_fee(margin)

        return {
            "lots": lots,
            "open_price": open_price,
            "units": lots * (contract_size or self.standard_lot),
            "margin_invested": margin,
            "brokerage_fee": fee,
            "initial_unrealized_pnl": -fee # Position starts negative by the fee amount
        }

    def calculate_pnl(self, position_type: str, lots: float, open_price: float, current_price: float, contract_size: float = None, is_base_usd: bool = False) -> float:
        """
        Calculates the Gross Profit and Loss (PnL) in USD.
        For USD-base pairs (USD/JPY, USD/CHF) the raw difference is in the QUOTE
        currency, so it's converted back to USD at the current price.
        """
        units = lots * (contract_size or self.standard_lot)

        if position_type.upper() == 'BUY':
            profit = (current_price - open_price) * units
        elif position_type.upper() == 'SELL':
            profit = (open_price - current_price) * units
        else:
            raise ValueError("Invalid position type. Must be BUY or SELL.")

        if is_base_usd and current_price != 0:
            profit = profit / current_price  # quote currency (JPY/CHF/CAD) -> USD

        return profit
