"""
Live Trading Bot Runner
Run automated trading bot with paper trading
"""
import sys
import argparse
from datetime import datetime

sys.path.append('.')

from trading_bot import (
    MomentumStrategy,
    MeanReversionStrategy,
    TradeExecutor
)


def main():
    parser = argparse.ArgumentParser(description='Run Trading Bot')
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['momentum', 'mean_reversion'],
        default='momentum',
        help='Trading strategy to use'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL'],
        help='Symbols to trade'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital for portfolio tracking'
    )
    
    args = parser.parse_args()
    
    # Create strategy
    if args.strategy == 'momentum':
        strategy = MomentumStrategy(
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
    elif args.strategy == 'mean_reversion':
        strategy = MeanReversionStrategy(
            bb_period=20,
            bb_std=2.0,
            rsi_period=14,
            rsi_threshold=50
        )
    else:
        print(f"Unknown strategy: {args.strategy}")
        return
    
    # Create executor
    executor = TradeExecutor(
        strategy=strategy,
        symbols=args.symbols,
        paper_trading=True,
        initial_capital=args.capital
    )
    
    print("\n" + "="*80)
    print(" "*25 + "TRADING BOT STARTING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Strategy: {strategy.name}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Check Interval: {args.interval}s ({args.interval/60:.1f} minutes)")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Mode: Paper Trading")
    print(f"\n⚠️  WARNING: This bot will execute REAL paper trades on Alpaca!")
    print("  Press Ctrl+C to stop the bot at any time.")
    print("="*80)
    
    response = input("\nStart trading bot? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Bot not started.")
        return
    
    # Run the bot
    try:
        executor.run(check_interval=args.interval)
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")
    finally:
        print("\n" + "="*80)
        print(" "*25 + "TRADING BOT STOPPED")
        print("="*80)


if __name__ == "__main__":
    main()
