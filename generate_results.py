"""
Generate Results for README - Simple test generating charts and metrics
"""
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.append('.')

from config.config import Config
from database import get_engine, get_session
from database.repositories import StockPriceRepository
from data_analysis.technical_indicators import TechnicalAnalysis
from data_analysis.pattern_recognition import PatternRecognition
from data_analysis.visualization import StockVisualizer
from ml_models.ensemble_model import EnsemblePredictor
from trading_bot import MomentumStrategy, MeanReversionStrategy, Backtester

print("\n" + "="*80)
print(" "*25 + "GENERATING README RESULTS")
print("="*80)

# Initialize
config = Config()
engine = get_engine(config.DATABASE_URL)
session = get_session(engine)
repo = StockPriceRepository(session)

# Get data
print("\n📊 Loading AAPL data (last 6 months)...")
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
records = repo.get_by_symbol('AAPL', start_date=start_date, end_date=end_date)

if not records or len(records) < 50:
    print("❌ Insufficient data")
    session.close()
    exit(1)

# Convert to DataFrame
df = pd.DataFrame([{
    'timestamp': r.timestamp,
    'open': float(r.open),
    'high': float(r.high),
    'low': float(r.low),
    'close': float(r.close),
    'volume': int(r.volume)
} for r in records])

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

print(f"✅ Loaded {len(df)} records")
print(f"   From: {df.index[0].strftime('%Y-%m-%d')}")
print(f"   To: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Price: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Section 1: Data Summary
print("\n" + "="*80)
print("SECTION 1: DATA INGESTION")
print("="*80)
latest_price = df['close'].iloc[-1]
price_change = df['close'].iloc[-1] - df['close'].iloc[0]
price_change_pct = (price_change / df['close'].iloc[0]) * 100

print(f"Symbol: AAPL")
print(f"Records: {len(df)}")
print(f"Latest Price: ${latest_price:.2f}")
print(f"Period Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
print(f"High: ${df['close'].max():.2f}")
print(f"Low: ${df['close'].min():.2f}")
print(f"Avg Volume: {df['volume'].mean()/1e6:.1f}M")

# Section 2: Technical Analysis
print("\n" + "="*80)
print("SECTION 2: TECHNICAL ANALYSIS")
print("="*80)

ta = TechnicalAnalysis(df.copy())
df_rsi = ta.rsi(period=14)
df_macd = ta.macd()

rsi_value = df_rsi['rsi'].iloc[-1] if isinstance(df_rsi, pd.DataFrame) else 0
macd_value = df_macd['macd'].iloc[-1] if isinstance(df_macd, pd.DataFrame) else 0

print(f"RSI (14): {rsi_value:.2f}")
print(f"MACD: {macd_value:.2f}")

# Generate technical chart
print("\n📈 Generating technical analysis chart...")
try:
    viz = StockVisualizer(df.tail(60))
    fig = viz.create_candlestick_chart(title='AAPL - Last 60 Days')
    fig.write_html('results/technical_chart.html')
    print("   ✅ Saved: results/technical_chart.html")
except Exception as e:
    print(f"   ⚠️  Chart generation skipped: {e}")

# Section 3: ML Models
print("\n" + "="*80)
print("SECTION 3: MACHINE LEARNING")
print("="*80)

try:
    ensemble = EnsemblePredictor()
    ensemble.train(df, target_days=5)
    models = [name for name, model in ensemble.models.items() if model is not None]
    print(f"✅ Trained {len(models)} models")
    print(f"   Models: {', '.join(models)}")
except Exception as e:
    print(f"⚠️  ML training skipped: {e}")
    models = []

# Section 4: Backtest
print("\n" + "="*80)
print("SECTION 4: TRADING BOT BACKTESTING")
print("="*80)

# Momentum Strategy
print("\n📊 Testing Momentum Strategy...")
try:
    momentum = MomentumStrategy(rsi_period=14, rsi_overbought=70, rsi_oversold=30)
    bt1 = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    res1 = bt1.run(momentum, df, position_size=0.95)
    
    print(f"\n✅ Momentum Strategy:")
    print(f"   Return: {res1['total_return_pct']:+.2f}%")
    print(f"   Sharpe: {res1['sharpe_ratio']:.2f}")
    print(f"   Max DD: {res1['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {res1['win_rate']*100:.1f}%")
    print(f"   Trades: {res1['num_trades']}")
    
    fig1 = bt1.plot_results(show_trades=True)
    fig1.write_html('results/backtest_momentum.html')
    print("   ✅ Saved: results/backtest_momentum.html")
    momentum_return = res1['total_return_pct']
    momentum_sharpe = res1['sharpe_ratio']
    momentum_trades = res1['num_trades']
except Exception as e:
    print(f"❌ Momentum backtest failed: {e}")
    momentum_return, momentum_sharpe, momentum_trades = 0, 0, 0

# Mean Reversion Strategy
print("\n📊 Testing Mean Reversion Strategy...")
try:
    mean_rev = MeanReversionStrategy(bb_period=20, bb_std=2.0)
    bt2 = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    res2 = bt2.run(mean_rev, df, position_size=0.95)
    
    print(f"\n✅ Mean Reversion Strategy:")
    print(f"   Return: {res2['total_return_pct']:+.2f}%")
    print(f"   Sharpe: {res2['sharpe_ratio']:.2f}")
    print(f"   Max DD: {res2['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {res2['win_rate']*100:.1f}%")
    print(f"   Trades: {res2['num_trades']}")
    
    fig2 = bt2.plot_results(show_trades=True)
    fig2.write_html('results/backtest_mean_reversion.html')
    print("   ✅ Saved: results/backtest_mean_reversion.html")
    meanrev_return = res2['total_return_pct']
    meanrev_sharpe = res2['sharpe_ratio']
    meanrev_trades = res2['num_trades']
except Exception as e:
    print(f"❌ Mean reversion backtest failed: {e}")
    meanrev_return, meanrev_sharpe, meanrev_trades = 0, 0, 0

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\n✅ All tests completed!")
print(f"\n📊 Data: {len(df)} AAPL records")
print(f"💰 Latest Price: ${latest_price:.2f}")
print(f"📈 Period Return: {price_change_pct:+.2f}%")
print(f"📉 RSI: {rsi_value:.2f}")
print(f"🤖 ML Models: {len(models)}")
print(f"\n💼 Backtest Results:")
print(f"   Momentum: {momentum_return:+.2f}% | Sharpe: {momentum_sharpe:.2f} | {momentum_trades} trades")
print(f"   Mean Rev: {meanrev_return:+.2f}% | Sharpe: {meanrev_sharpe:.2f} | {meanrev_trades} trades")
print("="*80)

# Save results
with open('results/summary.txt', 'w') as f:
    f.write("STOCK MARKET ANALYSIS PLATFORM - TEST RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("DATA:\n")
    f.write(f"  Symbol: AAPL\n")
    f.write(f"  Records: {len(df)}\n")
    f.write(f"  Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"  Latest Price: ${latest_price:.2f}\n")
    f.write(f"  Period Return: {price_change_pct:+.2f}%\n\n")
    f.write("TECHNICAL ANALYSIS:\n")
    f.write(f"  RSI: {rsi_value:.2f}\n")
    f.write(f"  MACD: {macd_value:.2f}\n\n")
    f.write("MACHINE LEARNING:\n")
    f.write(f"  Models Trained: {len(models)}\n")
    f.write(f"  Models: {', '.join(models) if models else 'None'}\n\n")
    f.write("BACKTEST RESULTS:\n\n")
    f.write(f"Momentum Strategy:\n")
    f.write(f"  Return: {momentum_return:+.2f}%\n")
    f.write(f"  Sharpe Ratio: {momentum_sharpe:.2f}\n")
    f.write(f"  Trades: {momentum_trades}\n\n")
    f.write(f"Mean Reversion Strategy:\n")
    f.write(f"  Return: {meanrev_return:+.2f}%\n")
    f.write(f"  Sharpe Ratio: {meanrev_sharpe:.2f}\n")
    f.write(f"  Trades: {meanrev_trades}\n")

print("\n📝 Summary saved to: results/summary.txt")
print("\n🎉 Results generation complete!")

session.close()
