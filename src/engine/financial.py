import yaml

class FinancialEngine:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.leverage = self.config['trading']['leverage']
        self.fee_pct = self.config['trading']['brokerage_fee_pct']
        self.standard_lot = self.config['trading']['standard_lot_size']

    def calculate_margin(self, lots: float, current_price: float, is_base_usd: bool = True) -> float:
        """
        Calculates the margin required to open a position.
        If base currency of the pair is USD (like USD/JPY), margin is lots * standard_lot / leverage.
        If base currency is not USD (like EUR/USD), margin is lots * standard_lot * current_price / leverage.
        For simplicity, assuming the provided price handles base conversion.
        """
        units = lots * self.standard_lot
        # Simplified: Units * Price / Leverage
        margin = (units * current_price) / self.leverage
        return margin

    def calculate_brokerage_fee(self, margin: float) -> float:
        """
        Brokerage fee is a percentage of the margin invested.
        """
        return margin * self.fee_pct

    def open_position_math(self, lots: float, open_price: float):
        """
        Returns a dictionary with margin and fee details when opening a position.
        """
        margin = self.calculate_margin(lots, open_price)
        fee = self.calculate_brokerage_fee(margin)
        
        return {
            "lots": lots,
            "open_price": open_price,
            "units": lots * self.standard_lot,
            "margin_invested": margin,
            "brokerage_fee": fee,
            "initial_unrealized_pnl": -fee # Position starts negative by the fee amount
        }

    def calculate_pnl(self, position_type: str, lots: float, open_price: float, current_price: float) -> float:
        """
        Calculates the Gross Profit and Loss (PnL) in USD.
        Simplified pip value calculation.
        """
        units = lots * self.standard_lot
        
        if position_type.upper() == 'BUY':
            profit = (current_price - open_price) * units
        elif position_type.upper() == 'SELL':
            profit = (open_price - current_price) * units
        else:
            raise ValueError("Invalid position type. Must be BUY or SELL.")
            
        return profit
