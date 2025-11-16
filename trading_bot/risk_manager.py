"""
Risk Management System for Trading Bot
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np


class RiskManager:
    """
    Manages risk parameters and position sizing for trading
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # Max 10% of portfolio per position
        max_portfolio_risk: float = 0.02,  # Max 2% of portfolio at risk
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.15,  # 15% take profit
        max_daily_loss: float = 0.05,  # Max 5% daily loss
        max_correlation: float = 0.7  # Max correlation between positions
    ):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk as fraction
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_daily_loss: Maximum daily loss as fraction
            max_correlation: Maximum correlation between positions
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.max_correlation = max_correlation
        
        self.daily_pnl = 0.0
        self.starting_capital = 0.0
    
    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        volatility: Optional[float] = None
    ) -> int:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            account_value: Current account value
            entry_price: Entry price for the position
            volatility: Asset volatility (optional, for Kelly criterion)
            
        Returns:
            Number of shares to trade
        """
        # Basic position sizing: don't risk more than max_portfolio_risk
        risk_amount = account_value * self.max_portfolio_risk
        shares_by_risk = int(risk_amount / (entry_price * self.stop_loss_pct))
        
        # Don't exceed max position size
        max_shares = int((account_value * self.max_position_size) / entry_price)
        
        # Take the minimum
        shares = min(shares_by_risk, max_shares)
        
        # Ensure at least 1 share if possible
        if shares == 0 and account_value > entry_price:
            shares = 1
        
        return max(0, shares)
    
    def calculate_stop_loss(self, entry_price: float, side: str = 'long') -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        if side == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, side: str = 'long') -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        if side == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:  # short
            return entry_price * (1 - self.take_profit_pct)
    
    def should_stop_trading(self, current_value: float) -> bool:
        """
        Check if daily loss limit has been reached
        
        Args:
            current_value: Current account value
            
        Returns:
            True if should stop trading for the day
        """
        if self.starting_capital == 0:
            self.starting_capital = current_value
            return False
        
        daily_loss = (self.starting_capital - current_value) / self.starting_capital
        return daily_loss >= self.max_daily_loss
    
    def reset_daily_tracking(self, current_value: float):
        """Reset daily P&L tracking"""
        self.starting_capital = current_value
        self.daily_pnl = 0.0
    
    def validate_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        account_value: float,
        current_positions: Dict
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if a trade should be executed
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Trade price
            account_value: Current account value
            current_positions: Dictionary of current positions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if daily loss limit reached
        if self.should_stop_trading(account_value):
            return False, "Daily loss limit reached"
        
        # Check position size
        position_value = quantity * price
        position_pct = position_value / account_value
        
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.1%} exceeds maximum {self.max_position_size:.1%}"
        
        # Check if already have position in symbol
        if symbol in current_positions:
            return False, f"Already have position in {symbol}"
        
        # Check total portfolio risk
        total_risk = sum(
            pos['quantity'] * pos['entry_price'] * self.stop_loss_pct
            for pos in current_positions.values()
        )
        total_risk += quantity * price * self.stop_loss_pct
        
        total_risk_pct = total_risk / account_value
        if total_risk_pct > self.max_portfolio_risk * len(current_positions) + self.max_portfolio_risk:
            return False, "Total portfolio risk too high"
        
        return True, None
    
    def calculate_portfolio_risk(self, positions: Dict, account_value: float) -> Dict:
        """
        Calculate current portfolio risk metrics
        
        Args:
            positions: Dictionary of current positions
            account_value: Current account value
            
        Returns:
            Dictionary with risk metrics
        """
        if not positions:
            return {
                'total_exposure': 0.0,
                'total_risk': 0.0,
                'risk_pct': 0.0,
                'concentration': {}
            }
        
        total_exposure = sum(
            pos['quantity'] * pos['current_price']
            for pos in positions.values()
        )
        
        total_risk = sum(
            pos['quantity'] * pos['entry_price'] * self.stop_loss_pct
            for pos in positions.values()
        )
        
        concentration = {
            symbol: (pos['quantity'] * pos['current_price']) / account_value
            for symbol, pos in positions.items()
        }
        
        return {
            'total_exposure': total_exposure,
            'total_risk': total_risk,
            'risk_pct': total_risk / account_value,
            'concentration': concentration
        }
