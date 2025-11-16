"""
Trading Bot Package
Implements automated trading strategies with Alpaca API integration
"""

from .alpaca_client import AlpacaClient
from .strategies import (
    TradingStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    MLStrategy
)
from .backtester import Backtester
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .trade_executor import TradeExecutor

__all__ = [
    'AlpacaClient',
    'TradingStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MLStrategy',
    'Backtester',
    'RiskManager',
    'PortfolioManager',
    'TradeExecutor'
]
