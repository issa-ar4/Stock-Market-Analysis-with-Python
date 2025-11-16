"""
Demo: Section 4 - Trading Bot
Demonstrates backtesting and paper trading capabilities
"""
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append('.')

from trading_bot import (
    MomentumStrategy,
    MeanReversionStrategy,
    MLStrategy,
    Backtester,
    RiskManager,
    PortfolioManager,
    AlpacaClient
)
from database import get_session, get_engine, StockPriceRepository
from ml_models.ensemble_model import EnsemblePredictor
from config.config import Config


def demo_backtesting():
    """Demonstrate strategy backtesting"""
    print("\n" + "="*80)
    print("DEMO 1: STRATEGY BACKTESTING")
    print("="*80)
    
    # Initialize database
    config = Config()
    engine = get_engine(config.DATABASE_URL)
    db = get_session(engine)
    repo = StockPriceRepository(db)
    
    # Get historical data for AAPL
    print("\n📊 Loading historical data for AAPL...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = repo.get_by_symbol('AAPL', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) < 50:
        print("❌ Insufficient data for backtesting. Please fetch more historical data.")
        print("   Run: python3 scripts/fetch_historical_data.py --symbol AAPL --period 1y")
        db.close()
        return
    
    print(f"✅ Loaded {len(df)} days of data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test 1: Momentum Strategy
    print("\n" + "-"*80)
    print("Test 1: Momentum Strategy (RSI + MACD)")
    print("-"*80)
    
    momentum_strategy = MomentumStrategy(
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30
    )
    
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    print("\n🔄 Running backtest...")
    results = backtester.run(momentum_strategy, df, position_size=0.95)
    backtester.print_summary()
    
    # Save plot
    print("📈 Generating backtest chart...")
    fig = backtester.plot_results(show_trades=True)
    fig.write_html('backtest_momentum.html')
    print("   Saved to: backtest_momentum.html")
    
    # Test 2: Mean Reversion Strategy
    print("\n" + "-"*80)
    print("Test 2: Mean Reversion Strategy (Bollinger Bands)")
    print("-"*80)
    
    mean_reversion_strategy = MeanReversionStrategy(
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_threshold=50
    )
    
    backtester2 = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    print("\n🔄 Running backtest...")
    results2 = backtester2.run(mean_reversion_strategy, df, position_size=0.95)
    backtester2.print_summary()
    
    # Save plot
    print("📈 Generating backtest chart...")
    fig2 = backtester2.plot_results(show_trades=True)
    fig2.write_html('backtest_mean_reversion.html')
    print("   Saved to: backtest_mean_reversion.html")
    
    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<25} {'Return':>12} {'Sharpe':>10} {'Max DD':>10} {'Win Rate':>10}")
    print("-"*80)
    print(f"{'Momentum':<25} {results['total_return_pct']:>11.2f}% {results['sharpe_ratio']:>10.2f} "
          f"{results['max_drawdown_pct']:>9.2f}% {results['win_rate']*100:>9.1f}%")
    print(f"{'Mean Reversion':<25} {results2['total_return_pct']:>11.2f}% {results2['sharpe_ratio']:>10.2f} "
          f"{results2['max_drawdown_pct']:>9.2f}% {results2['win_rate']*100:>9.1f}%")
    print("="*80)
    
    db.close()


def demo_ml_strategy():
    """Demonstrate ML-based strategy"""
    print("\n" + "="*80)
    print("DEMO 2: ML-BASED TRADING STRATEGY")
    print("="*80)
    
    # Initialize database
    config = Config()
    engine = get_engine(config.DATABASE_URL)
    db = get_session(engine)
    repo = StockPriceRepository(db)
    
    # Get historical data
    print("\n📊 Loading historical data for AAPL...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = repo.get_by_symbol('AAPL', start_date=start_date, end_date=end_date)
    
    if df is None or len(df) < 100:
        print("❌ Insufficient data for ML strategy. Need at least 1 year of data.")
        print("   Run: python3 scripts/fetch_historical_data.py --symbol AAPL --period 1y")
        db.close()
        return
    
    print(f"✅ Loaded {len(df)} days of data")
    
    # Train ML models
    print("\n🤖 Training ensemble ML models...")
    try:
        ensemble = EnsemblePredictor()
        ensemble.train(df, target_days=5)
        
        models = ensemble.models
        print(f"✅ Trained {len([m for m in models.values() if m is not None])} models")
        
        # Create ML strategy
        ml_strategy = MLStrategy(
            model_dict=models,
            confidence_threshold=0.6,
            lookback_period=60
        )
        
        # Backtest ML strategy
        print("\n🔄 Running backtest with ML strategy...")
        backtester = Backtester(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        # Use recent 6 months for testing
        test_df = df.tail(126)  # ~6 months
        results = backtester.run(ml_strategy, test_df, position_size=0.95)
        backtester.print_summary()
        
        # Save plot
        print("📈 Generating backtest chart...")
        fig = backtester.plot_results(show_trades=True)
        fig.write_html('backtest_ml_strategy.html')
        print("   Saved to: backtest_ml_strategy.html")
        
    except Exception as e:
        print(f"❌ Error training ML models: {e}")
    
    db.close()


def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "="*80)
    print("DEMO 3: RISK MANAGEMENT")
    print("="*80)
    
    risk_manager = RiskManager(
        max_position_size=0.1,      # Max 10% per position
        max_portfolio_risk=0.02,    # Max 2% risk
        stop_loss_pct=0.05,         # 5% stop loss
        take_profit_pct=0.15,       # 15% take profit
        max_daily_loss=0.05         # 5% max daily loss
    )
    
    # Example calculations
    account_value = 100000
    entry_price = 150.0
    
    print("\n📊 Risk Parameters:")
    print(f"   Max Position Size: {risk_manager.max_position_size*100:.1f}%")
    print(f"   Max Portfolio Risk: {risk_manager.max_portfolio_risk*100:.1f}%")
    print(f"   Stop Loss: {risk_manager.stop_loss_pct*100:.1f}%")
    print(f"   Take Profit: {risk_manager.take_profit_pct*100:.1f}%")
    print(f"   Max Daily Loss: {risk_manager.max_daily_loss*100:.1f}%")
    
    print("\n📈 Position Sizing Example:")
    print(f"   Account Value: ${account_value:,.2f}")
    print(f"   Entry Price: ${entry_price:.2f}")
    
    position_size = risk_manager.calculate_position_size(account_value, entry_price)
    position_value = position_size * entry_price
    position_pct = (position_value / account_value) * 100
    
    print(f"   → Position Size: {position_size} shares")
    print(f"   → Position Value: ${position_value:,.2f} ({position_pct:.1f}%)")
    
    stop_loss_price = risk_manager.calculate_stop_loss(entry_price, 'long')
    take_profit_price = risk_manager.calculate_take_profit(entry_price, 'long')
    
    print(f"   → Stop Loss: ${stop_loss_price:.2f} ({-risk_manager.stop_loss_pct*100:.1f}%)")
    print(f"   → Take Profit: ${take_profit_price:.2f} ({+risk_manager.take_profit_pct*100:.1f}%)")
    
    # Portfolio risk calculation
    print("\n📊 Portfolio Risk Example:")
    positions = {
        'AAPL': {
            'quantity': 50,
            'entry_price': 150.0,
            'current_price': 155.0
        },
        'MSFT': {
            'quantity': 30,
            'entry_price': 350.0,
            'current_price': 345.0
        }
    }
    
    risk_metrics = risk_manager.calculate_portfolio_risk(positions, account_value)
    
    print(f"   Total Exposure: ${risk_metrics['total_exposure']:,.2f}")
    print(f"   Total Risk: ${risk_metrics['total_risk']:,.2f} ({risk_metrics['risk_pct']*100:.2f}%)")
    print(f"   Concentration:")
    for symbol, pct in risk_metrics['concentration'].items():
        print(f"      {symbol}: {pct*100:.1f}%")


def demo_paper_trading():
    """Demonstrate paper trading with Alpaca"""
    print("\n" + "="*80)
    print("DEMO 4: ALPACA PAPER TRADING")
    print("="*80)
    
    try:
        alpaca = AlpacaClient(paper_trading=True)
        
        # Get account info
        print("\n📊 Account Information:")
        account = alpaca.get_account()
        
        print(f"   Account Status: {account['status']}")
        print(f"   Buying Power: ${float(account['buying_power']):,.2f}")
        print(f"   Cash: ${float(account['cash']):,.2f}")
        print(f"   Portfolio Value: ${float(account['portfolio_value']):,.2f}")
        print(f"   Pattern Day Trader: {account['pattern_day_trader']}")
        
        # Get positions
        print("\n📈 Current Positions:")
        positions = alpaca.get_positions()
        
        if positions:
            for pos in positions:
                pnl = float(pos['unrealized_pl'])
                pnl_pct = float(pos['unrealized_plpc']) * 100
                print(f"   {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']} "
                      f"(P&L: ${pnl:.2f}, {pnl_pct:+.2f}%)")
        else:
            print("   No open positions")
        
        # Check market status
        print("\n⏰ Market Status:")
        is_open = alpaca.is_market_open()
        print(f"   Market is {'OPEN 📈' if is_open else 'CLOSED 📴'}")
        
        # Get calendar
        calendar = alpaca.get_calendar()
        if calendar:
            next_day = calendar[0]
            print(f"   Next Trading Day: {next_day['date']}")
            print(f"   Session: {next_day['open']} - {next_day['close']}")
        
        # Recent orders
        print("\n📋 Recent Orders:")
        orders = alpaca.get_orders(status='all')
        
        if orders:
            for order in orders[:5]:  # Show last 5 orders
                print(f"   {order['created_at'][:10]}: {order['side'].upper()} {order['qty']} "
                      f"{order['symbol']} @ {order['type']} - {order['status']}")
        else:
            print("   No recent orders")
        
        print("\n✅ Successfully connected to Alpaca paper trading account!")
        print("   You can now run the live trading bot with this account.")
        
    except Exception as e:
        print(f"\n❌ Error connecting to Alpaca: {e}")
        print("\nPlease check:")
        print("1. Your API keys are correct in .env file")
        print("2. You have internet connection")
        print("3. Your Alpaca paper trading account is active")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print(" "*25 + "SECTION 4: TRADING BOT DEMO")
    print("="*80)
    
    # Demo 1: Backtesting
    demo_backtesting()
    
    # Demo 2: ML Strategy
    print("\n" + "="*80)
    print("Would you like to test ML strategy? (requires 1 year of data)")
    response = input("Continue? (y/n): ").strip().lower()
    if response == 'y':
        demo_ml_strategy()
    
    # Demo 3: Risk Management
    demo_risk_management()
    
    # Demo 4: Paper Trading
    print("\n" + "="*80)
    print("Would you like to test Alpaca paper trading connection?")
    response = input("Continue? (y/n): ").strip().lower()
    if response == 'y':
        demo_paper_trading()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review backtest results in the generated HTML files")
    print("2. Adjust strategy parameters based on backtest results")
    print("3. Run live paper trading bot: python3 run_trading_bot.py")
    print("4. Monitor performance and refine strategies")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
