"""
Portfolio Manager for Trading Bot
"""
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class PortfolioManager:
    """
    Manages portfolio positions and tracks performance
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> position info
        self.trade_history = []
        self.portfolio_history = []
        
        # Track performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value (cash + positions)
        """
        position_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def get_available_cash(self) -> float:
        """Get available cash for trading"""
        return self.cash
    
    def open_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        timestamp: datetime
    ) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            timestamp: Trade timestamp
            
        Returns:
            True if position opened successfully
        """
        cost = quantity * entry_price
        
        if cost > self.cash:
            return False
        
        if symbol in self.positions:
            # Add to existing position
            existing = self.positions[symbol]
            total_quantity = existing['quantity'] + quantity
            total_cost = (existing['quantity'] * existing['entry_price']) + cost
            avg_price = total_cost / total_quantity
            
            self.positions[symbol] = {
                'quantity': total_quantity,
                'entry_price': avg_price,
                'entry_time': existing['entry_time'],
                'last_update': timestamp
            }
        else:
            # Create new position
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'last_update': timestamp
            }
        
        self.cash -= cost
        
        # Record trade
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'type': 'BUY',
            'quantity': quantity,
            'price': entry_price,
            'cost': cost,
            'cash_remaining': self.cash
        })
        
        self.total_trades += 1
        
        return True
    
    def close_position(
        self,
        symbol: str,
        quantity: Optional[int],
        exit_price: float,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Close a position (fully or partially)
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell (None = all)
            exit_price: Exit price
            timestamp: Trade timestamp
            
        Returns:
            Trade information or None if failed
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if quantity is None:
            quantity = position['quantity']
        
        if quantity > position['quantity']:
            quantity = position['quantity']
        
        # Calculate P&L
        proceeds = quantity * exit_price
        cost_basis = quantity * position['entry_price']
        profit = proceeds - cost_basis
        
        self.cash += proceeds
        
        # Update position
        if quantity == position['quantity']:
            # Close entire position
            del self.positions[symbol]
        else:
            # Partial close
            position['quantity'] -= quantity
            position['last_update'] = timestamp
        
        # Update statistics
        if profit > 0:
            self.winning_trades += 1
            self.total_profit += profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit)
        
        # Record trade
        trade_info = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': 'SELL',
            'quantity': quantity,
            'price': exit_price,
            'proceeds': proceeds,
            'profit': profit,
            'profit_pct': (profit / cost_basis) * 100,
            'cash_remaining': self.cash
        }
        
        self.trade_history.append(trade_info)
        self.total_trades += 1
        
        return trade_info
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position information for a symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Get all current positions"""
        return self.positions.copy()
    
    def has_position(self, symbol: str) -> bool:
        """Check if holding position in symbol"""
        return symbol in self.positions
    
    def get_position_pnl(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict]:
        """
        Get unrealized P&L for a position
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            Dictionary with P&L information
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_value = position['quantity'] * current_price
        cost_basis = position['quantity'] * position['entry_price']
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
        
        return {
            'symbol': symbol,
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'current_price': current_price,
            'cost_basis': cost_basis,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get comprehensive portfolio summary
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary with portfolio metrics
        """
        total_value = self.get_portfolio_value(current_prices)
        total_return = total_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position['quantity'] * current_prices[symbol]
                cost_basis = position['quantity'] * position['entry_price']
                unrealized_pnl += (current_value - cost_basis)
        
        # Realized P&L
        realized_pnl = self.total_profit - self.total_loss
        
        # Win rate
        total_closed_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'num_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate
        }
    
    def print_portfolio_summary(self, current_prices: Dict[str, float]):
        """Print portfolio summary to console"""
        summary = self.get_portfolio_summary(current_prices)
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Initial Capital:     ${summary['initial_capital']:,.2f}")
        print(f"Cash:                ${summary['cash']:,.2f}")
        print(f"Total Value:         ${summary['total_value']:,.2f}")
        print(f"Total Return:        ${summary['total_return']:,.2f} ({summary['total_return_pct']:.2f}%)")
        print(f"Realized P&L:        ${summary['realized_pnl']:,.2f}")
        print(f"Unrealized P&L:      ${summary['unrealized_pnl']:,.2f}")
        print(f"Total P&L:           ${summary['total_pnl']:,.2f}")
        print(f"\nPositions:           {summary['num_positions']}")
        print(f"Total Trades:        {summary['total_trades']}")
        print(f"Winning Trades:      {summary['winning_trades']}")
        print(f"Losing Trades:       {summary['losing_trades']}")
        print(f"Win Rate:            {summary['win_rate']:.1f}%")
        print("="*60)
        
        # Print individual positions
        if self.positions:
            print("\nCURRENT POSITIONS")
            print("-"*60)
            for symbol in self.positions:
                pnl = self.get_position_pnl(symbol, current_prices.get(symbol, 0))
                if pnl:
                    print(f"{symbol:6s} | Qty: {pnl['quantity']:4d} | "
                          f"Entry: ${pnl['entry_price']:7.2f} | "
                          f"Current: ${pnl['current_price']:7.2f} | "
                          f"P&L: ${pnl['unrealized_pnl']:8.2f} ({pnl['unrealized_pnl_pct']:+.1f}%)")
            print("="*60 + "\n")
    
    def export_trade_history(self) -> pd.DataFrame:
        """Export trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
