"""
Trade Executor - Orchestrates Trading Bot Operations
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import pandas as pd
from sqlalchemy.orm import Session

from .alpaca_client import AlpacaClient
from .strategies import TradingStrategy
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from database import get_session, get_engine, StockPriceRepository
from config import config


class TradeExecutor:
    """
    Main trading bot executor
    Coordinates strategy, risk management, and order execution
    """
    
    def __init__(
        self,
        strategy: TradingStrategy,
        symbols: List[str],
        paper_trading: bool = True,
        initial_capital: float = 100000
    ):
        """
        Initialize trade executor
        
        Args:
            strategy: Trading strategy to use
            symbols: List of symbols to trade
            paper_trading: Use paper trading account
            initial_capital: Starting capital (for paper trading)
        """
        self.strategy = strategy
        self.symbols = symbols
        self.paper_trading = paper_trading
        
        # Initialize components
        self.alpaca = AlpacaClient(paper_trading=paper_trading)
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        
        # Database
        engine = get_engine(config.DATABASE_URL)
        self.db = get_session(engine)
        self.repository = StockPriceRepository(self.db)
        
        # State tracking
        self.is_running = False
        self.last_check_time = None
    
    def sync_portfolio_with_alpaca(self):
        """Sync portfolio manager with Alpaca account"""
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            
            # Update cash
            self.portfolio_manager.cash = float(account['cash'])
            
            # Update positions
            self.portfolio_manager.positions = {}
            for pos in positions:
                symbol = pos['symbol']
                self.portfolio_manager.positions[symbol] = {
                    'quantity': int(pos['qty']),
                    'entry_price': float(pos['avg_entry_price']),
                    'entry_time': datetime.now(),
                    'last_update': datetime.now()
                }
            
            print(f"✅ Portfolio synced with Alpaca")
            print(f"   Cash: ${self.portfolio_manager.cash:,.2f}")
            print(f"   Positions: {len(self.portfolio_manager.positions)}")
            
        except Exception as e:
            print(f"❌ Error syncing portfolio: {e}")
    
    def get_latest_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get latest data for a symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get data from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.repository.get_by_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 20:
                print(f"⚠️  Insufficient data for {symbol}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return None
    
    def check_signals(self) -> Dict[str, int]:
        """
        Check trading signals for all symbols
        
        Returns:
            Dictionary of symbol -> signal (1=buy, -1=sell, 0=hold)
        """
        signals = {}
        
        for symbol in self.symbols:
            try:
                df = self.get_latest_data(symbol)
                if df is not None:
                    signal = self.strategy.get_current_signal(df)
                    signals[symbol] = signal
                else:
                    signals[symbol] = 0
            except Exception as e:
                print(f"❌ Error checking signal for {symbol}: {e}")
                signals[symbol] = 0
        
        return signals
    
    def execute_buy(self, symbol: str, current_price: float) -> bool:
        """
        Execute buy order
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            True if order executed successfully
        """
        try:
            # Get account value
            account = self.alpaca.get_account()
            account_value = float(account['portfolio_value'])
            
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(
                account_value=account_value,
                entry_price=current_price
            )
            
            if quantity == 0:
                print(f"⚠️  Position size too small for {symbol}")
                return False
            
            # Validate trade
            current_positions = self.portfolio_manager.get_all_positions()
            is_valid, error_msg = self.risk_manager.validate_trade(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                account_value=account_value,
                current_positions=current_positions
            )
            
            if not is_valid:
                print(f"⚠️  Trade validation failed: {error_msg}")
                return False
            
            # Place order with Alpaca
            order = self.alpaca.place_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                order_type='market',
                time_in_force='day'
            )
            
            print(f"✅ BUY order placed: {symbol} x{quantity} @ ${current_price:.2f}")
            print(f"   Order ID: {order['id']}")
            
            # Update portfolio manager
            self.portfolio_manager.open_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=current_price,
                timestamp=datetime.now()
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing buy for {symbol}: {e}")
            return False
    
    def execute_sell(self, symbol: str, current_price: float) -> bool:
        """
        Execute sell order
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            True if order executed successfully
        """
        try:
            # Check if we have a position
            position = self.portfolio_manager.get_position(symbol)
            if position is None:
                print(f"⚠️  No position to sell for {symbol}")
                return False
            
            quantity = position['quantity']
            
            # Place order with Alpaca
            order = self.alpaca.place_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                order_type='market',
                time_in_force='day'
            )
            
            print(f"✅ SELL order placed: {symbol} x{quantity} @ ${current_price:.2f}")
            print(f"   Order ID: {order['id']}")
            
            # Update portfolio manager
            trade_info = self.portfolio_manager.close_position(
                symbol=symbol,
                quantity=quantity,
                exit_price=current_price,
                timestamp=datetime.now()
            )
            
            if trade_info:
                print(f"   P&L: ${trade_info['profit']:.2f} ({trade_info['profit_pct']:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing sell for {symbol}: {e}")
            return False
    
    def check_stop_loss_take_profit(self):
        """Check and execute stop loss / take profit orders"""
        for symbol in list(self.portfolio_manager.positions.keys()):
            try:
                position = self.portfolio_manager.get_position(symbol)
                if position is None:
                    continue
                
                # Get current price
                quote = self.alpaca.get_latest_quote(symbol)
                current_price = float(quote.get('ap', quote.get('bp', position['entry_price'])))
                
                # Calculate P&L
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check stop loss
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    print(f"🛑 Stop loss triggered for {symbol} ({pnl_pct*100:.2f}%)")
                    self.execute_sell(symbol, current_price)
                
                # Check take profit
                elif pnl_pct >= self.risk_manager.take_profit_pct:
                    print(f"🎯 Take profit triggered for {symbol} ({pnl_pct*100:.2f}%)")
                    self.execute_sell(symbol, current_price)
                    
            except Exception as e:
                print(f"❌ Error checking stop loss/take profit for {symbol}: {e}")
    
    def run_once(self):
        """Run one iteration of the trading loop"""
        print(f"\n{'='*60}")
        print(f"Trading Bot Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Check if market is open
        if not self.alpaca.is_market_open():
            print("📴 Market is closed")
            return
        
        print("📈 Market is open - checking signals...")
        
        # Check stop loss / take profit first
        self.check_stop_loss_take_profit()
        
        # Check signals for all symbols
        signals = self.check_signals()
        
        for symbol, signal in signals.items():
            try:
                # Get current price
                quote = self.alpaca.get_latest_quote(symbol)
                current_price = float(quote.get('ap', quote.get('bp', 0)))
                
                if current_price == 0:
                    continue
                
                has_position = self.portfolio_manager.has_position(symbol)
                
                # Execute trades based on signals
                if signal == 1 and not has_position:
                    # Buy signal
                    print(f"\n🟢 BUY signal for {symbol}")
                    self.execute_buy(symbol, current_price)
                
                elif signal == -1 and has_position:
                    # Sell signal
                    print(f"\n🔴 SELL signal for {symbol}")
                    self.execute_sell(symbol, current_price)
                
                elif signal == 0:
                    print(f"⚪ HOLD signal for {symbol} @ ${current_price:.2f}")
                    
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
        
        # Print portfolio summary
        current_prices = {}
        for symbol in self.symbols:
            try:
                quote = self.alpaca.get_latest_quote(symbol)
                current_prices[symbol] = float(quote.get('ap', quote.get('bp', 0)))
            except:
                pass
        
        self.portfolio_manager.print_portfolio_summary(current_prices)
        self.last_check_time = datetime.now()
    
    def run(self, check_interval: int = 60):
        """
        Run trading bot continuously
        
        Args:
            check_interval: Seconds between checks
        """
        print(f"\n🤖 Starting Trading Bot")
        print(f"Strategy: {self.strategy.name}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Paper Trading: {self.paper_trading}")
        print(f"Check Interval: {check_interval}s")
        
        # Sync portfolio
        self.sync_portfolio_with_alpaca()
        
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_once()
                
                # Wait before next check
                print(f"\n⏳ Waiting {check_interval}s until next check...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.db:
            self.db.close()
        print("Bot stopped.")
