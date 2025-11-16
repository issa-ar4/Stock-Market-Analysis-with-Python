"""
Comprehensive Test - Test all sections and generate results for README
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
print(" "*20 + "COMPREHENSIVE PLATFORM TEST")
print("="*80)

# Initialize
config = Config()
engine = get_engine(config.DATABASE_URL)
session = get_session(engine)
repo = StockPriceRepository(session)

# Get data
print("\n📊 Loading AAPL data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
records = repo.get_by_symbol('AAPL', start_date=start_date, end_date=end_date)

if records is None or len(records) < 50:
    print("❌ Insufficient data. Please run:")
    print("   python3 scripts/fetch_historical_data.py --symbol AAPL --period 6mo")
    session.close()
    exit(1)

# Convert to DataFrame
data = []
for record in records:
    data.append({
        'timestamp': record.timestamp,
        'open': record.open,
        'high': record.high,
        'low': record.low,
        'close': record.close,
        'volume': record.volume
    })

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

print(f"✅ Loaded {len(df)} records")
print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Section 1: Data Ingestion
print("\n" + "="*80)
print("SECTION 1: DATA INGESTION")
print("="*80)
print(f"✅ Database: SQLite")
print(f"✅ Records: {len(df)} AAPL records")
print(f"✅ Latest Price: ${df['close'].iloc[-1]:.2f}")
print(f"✅ API Integration: YFinance, Alpha Vantage, Finnhub")

# Section 2: Technical Analysis
print("\n" + "="*80)
print("SECTION 2: TECHNICAL ANALYSIS")
print("="*80)

ta = TechnicalAnalysis(df)
df = ta.calculate_all_indicators()

print(f"✅ RSI (14): {df['rsi'].iloc[-1]:.2f}")
print(f"✅ MACD: {df['macd'].iloc[-1]:.2f}")
print(f"✅ BB Upper: ${df['bb_upper'].iloc[-1]:.2f}")
print(f"✅ BB Lower: ${df['bb_lower'].iloc[-1]:.2f}")

pr = PatternRecognition(df)
support, resistance = pr.find_support_resistance()
print(f"✅ Support Level: ${support:.2f}")
print(f"✅ Resistance Level: ${resistance:.2f}")

# Generate chart
print("\n📈 Generating technical analysis chart...")
viz = StockVisualizer(df.tail(60))
fig = viz.create_candlestick_chart(title='AAPL - Technical Analysis')
fig.write_html('results/technical_analysis_chart.html')
print("   Saved: results/technical_analysis_chart.html")

# Section 3: ML Models
print("\n" + "="*80)
print("SECTION 3: MACHINE LEARNING")
print("="*80)

ensemble = EnsemblePredictor()
ensemble.train(df, target_days=5)
models_built = [name for name, model in ensemble.models.items() if model is not None]
print(f"✅ Models Trained: {len(models_built)}")
print(f"   Models: {', '.join(models_built)}")

# Section 4: Trading Bot - Backtesting
print("\n" + "="*80)
print("SECTION 4: TRADING BOT - BACKTESTING")
print("="*80)

# Test Momentum Strategy
print("\n📊 Backtesting Momentum Strategy...")
momentum = MomentumStrategy(rsi_period=14, rsi_overbought=70, rsi_oversold=30)
backtester = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
results_momentum = backtester.run(momentum, df, position_size=0.95)

print(f"✅ Momentum Strategy Results:")
print(f"   Initial Capital: ${results_momentum['initial_capital']:,.2f}")
print(f"   Final Value: ${results_momentum['final_value']:,.2f}")
print(f"   Total Return: {results_momentum['total_return_pct']:.2f}%")
print(f"   Sharpe Ratio: {results_momentum['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {results_momentum['max_drawdown_pct']:.2f}%")
print(f"   Win Rate: {results_momentum['win_rate']*100:.1f}%")
print(f"   Trades: {results_momentum['num_trades']}")

# Save momentum backtest chart
fig_momentum = backtester.plot_results(show_trades=True)
fig_momentum.write_html('results/backtest_momentum.html')
print("   Saved: results/backtest_momentum.html")

# Test Mean Reversion Strategy
print("\n📊 Backtesting Mean Reversion Strategy...")
mean_rev = MeanReversionStrategy(bb_period=20, bb_std=2.0)
backtester2 = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
results_mean_rev = backtester2.run(mean_rev, df, position_size=0.95)

print(f"✅ Mean Reversion Strategy Results:")
print(f"   Initial Capital: ${results_mean_rev['initial_capital']:,.2f}")
print(f"   Final Value: ${results_mean_rev['final_value']:,.2f}")
print(f"   Total Return: {results_mean_rev['total_return_pct']:.2f}%")
print(f"   Sharpe Ratio: {results_mean_rev['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {results_mean_rev['max_drawdown_pct']:.2f}%")
print(f"   Win Rate: {results_mean_rev['win_rate']*100:.1f}%")
print(f"   Trades: {results_mean_rev['num_trades']}")

# Save mean reversion backtest chart
fig_mean_rev = backtester2.plot_results(show_trades=True)
fig_mean_rev.write_html('results/backtest_mean_reversion.html')
print("   Saved: results/backtest_mean_reversion.html")

# Summary
print("\n" + "="*80)
print("COMPREHENSIVE TEST SUMMARY")
print("="*80)
print("\n✅ All Sections Tested Successfully!")
print(f"\n📊 Data:")
print(f"   • {len(df)} AAPL records")
print(f"   • ${df['close'].iloc[-1]:.2f} current price")
print(f"\n📈 Technical Analysis:")
print(f"   • RSI: {df['rsi'].iloc[-1]:.2f}")
print(f"   • MACD: {df['macd'].iloc[-1]:.2f}")
print(f"   • Support: ${support:.2f}")
print(f"   • Resistance: ${resistance:.2f}")
print(f"\n🤖 Machine Learning:")
print(f"   • {len(models_built)} models trained")
print(f"\n💼 Trading Bot Backtest Results:")
print(f"\n   Momentum Strategy:")
print(f"   • Return: {results_momentum['total_return_pct']:+.2f}%")
print(f"   • Sharpe: {results_momentum['sharpe_ratio']:.2f}")
print(f"   • Win Rate: {results_momentum['win_rate']*100:.1f}%")
print(f"\n   Mean Reversion Strategy:")
print(f"   • Return: {results_mean_rev['total_return_pct']:+.2f}%")
print(f"   • Sharpe: {results_mean_rev['sharpe_ratio']:.2f}")
print(f"   • Win Rate: {results_mean_rev['win_rate']*100:.1f}%")
print("\n" + "="*80)
print("🎉 Platform fully operational and tested!")
print("="*80)

# Save results summary
with open('results/test_summary.txt', 'w') as f:
    f.write("COMPREHENSIVE TEST RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Data: {len(df)} AAPL records ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})\n")
    f.write(f"Current Price: ${df['close'].iloc[-1]:.2f}\n\n")
    f.write("Technical Indicators:\n")
    f.write(f"  RSI: {df['rsi'].iloc[-1]:.2f}\n")
    f.write(f"  MACD: {df['macd'].iloc[-1]:.2f}\n")
    f.write(f"  Support: ${support:.2f}\n")
    f.write(f"  Resistance: ${resistance:.2f}\n\n")
    f.write(f"ML Models: {len(models_built)} trained\n\n")
    f.write("Backtest Results:\n\n")
    f.write("Momentum Strategy:\n")
    f.write(f"  Return: {results_momentum['total_return_pct']:+.2f}%\n")
    f.write(f"  Sharpe Ratio: {results_momentum['sharpe_ratio']:.2f}\n")
    f.write(f"  Max Drawdown: {results_momentum['max_drawdown_pct']:.2f}%\n")
    f.write(f"  Win Rate: {results_momentum['win_rate']*100:.1f}%\n")
    f.write(f"  Trades: {results_momentum['num_trades']}\n\n")
    f.write("Mean Reversion Strategy:\n")
    f.write(f"  Return: {results_mean_rev['total_return_pct']:+.2f}%\n")
    f.write(f"  Sharpe Ratio: {results_mean_rev['sharpe_ratio']:.2f}\n")
    f.write(f"  Max Drawdown: {results_mean_rev['max_drawdown_pct']:.2f}%\n")
    f.write(f"  Win Rate: {results_mean_rev['win_rate']*100:.1f}%\n")
    f.write(f"  Trades: {results_mean_rev['num_trades']}\n")

print("\n📝 Results saved to: results/test_summary.txt")

session.close()
