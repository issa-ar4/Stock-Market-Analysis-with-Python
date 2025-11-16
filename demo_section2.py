#!/usr/bin/env python
"""
Demo script showcasing Section 2: Data Analysis and Visualization
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion import StockDataFetcher
from data_analysis import TechnicalAnalysis, PatternRecognition, StockVisualizer, create_summary_table
from utils.logger import logger
from datetime import datetime, timedelta


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_technical_indicators():
    """Demonstrate technical indicator calculations"""
    print_header("Technical Indicators Demo")
    
    # Fetch data
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Fetching 90 days of data for {symbol}...\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )
    
    if df.empty:
        print("No data available")
        return
    
    # Initialize technical analysis
    ta = TechnicalAnalysis(df)
    
    # Calculate various indicators
    print("Calculating technical indicators...\n")
    
    # Moving Averages
    sma_20 = ta.sma(20)
    sma_50 = ta.sma(50)
    ema_12 = ta.ema(12)
    
    print(f"Moving Averages (Latest):")
    print(f"  SMA(20):  ${sma_20.iloc[-1]:.2f}")
    print(f"  SMA(50):  ${sma_50.iloc[-1]:.2f}")
    print(f"  EMA(12):  ${ema_12.iloc[-1]:.2f}")
    
    # RSI
    rsi = ta.rsi(14)
    print(f"\nRSI(14):    {rsi.iloc[-1]:.2f}")
    
    if rsi.iloc[-1] < 30:
        print("  ⚠️  Status: OVERSOLD - Potential buy signal")
    elif rsi.iloc[-1] > 70:
        print("  ⚠️  Status: OVERBOUGHT - Potential sell signal")
    else:
        print("  ✓ Status: NEUTRAL")
    
    # MACD
    macd_line, signal_line, histogram = ta.macd()
    print(f"\nMACD:")
    print(f"  MACD Line:   {macd_line.iloc[-1]:.2f}")
    print(f"  Signal Line: {signal_line.iloc[-1]:.2f}")
    print(f"  Histogram:   {histogram.iloc[-1]:.2f}")
    
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        print("  ✓ Status: BULLISH")
    else:
        print("  ⚠️  Status: BEARISH")
    
    # Bollinger Bands
    upper, middle, lower = ta.bollinger_bands(20, 2.0)
    current_price = df['close'].iloc[-1]
    
    print(f"\nBollinger Bands:")
    print(f"  Upper Band:  ${upper.iloc[-1]:.2f}")
    print(f"  Middle Band: ${middle.iloc[-1]:.2f}")
    print(f"  Lower Band:  ${lower.iloc[-1]:.2f}")
    print(f"  Current:     ${current_price:.2f}")
    
    if current_price >= upper.iloc[-1]:
        print("  ⚠️  Price at upper band - Potential resistance")
    elif current_price <= lower.iloc[-1]:
        print("  ⚠️  Price at lower band - Potential support")
    else:
        print("  ✓ Price within bands")
    
    # ATR (Volatility)
    atr = ta.atr(14)
    print(f"\nATR(14):    ${atr.iloc[-1]:.2f}")
    print(f"  Volatility: {(atr.iloc[-1] / current_price * 100):.2f}%")
    
    # Volume indicators
    obv = ta.obv()
    mfi = ta.mfi(14)
    
    print(f"\nVolume Indicators:")
    print(f"  OBV:        {obv.iloc[-1]:,.0f}")
    print(f"  MFI(14):    {mfi.iloc[-1]:.2f}")
    
    # Get all signals
    print("\n" + "-" * 70)
    print("Trading Signals:")
    signals = ta.get_trading_signals()
    latest_signals = signals.iloc[-1]
    
    if latest_signals['RSI_OVERSOLD']:
        print("  🟢 RSI: Oversold - Potential buy opportunity")
    if latest_signals['RSI_OVERBOUGHT']:
        print("  🔴 RSI: Overbought - Potential sell opportunity")
    
    if latest_signals['MACD_BULLISH']:
        print("  🟢 MACD: Bullish crossover detected")
    if latest_signals['MACD_BEARISH']:
        print("  🔴 MACD: Bearish crossover detected")
    
    if latest_signals['MA_GOLDEN_CROSS']:
        print("  🟢 MA: Golden cross (bullish)")
    if latest_signals['MA_DEATH_CROSS']:
        print("  🔴 MA: Death cross (bearish)")
    
    if latest_signals['BB_LOWER_TOUCH']:
        print("  🟢 BB: Price touching lower band")
    if latest_signals['BB_UPPER_TOUCH']:
        print("  🔴 BB: Price touching upper band")
    
    if not latest_signals.any():
        print("  ℹ️  No significant signals detected")
    
    fetcher.close()


def demo_pattern_recognition():
    """Demonstrate pattern recognition"""
    print_header("Pattern Recognition Demo")
    
    # Fetch data
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Analyzing patterns for {symbol}...\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )
    
    if df.empty:
        print("No data available")
        return
    
    # Initialize pattern recognition
    pr = PatternRecognition(df)
    
    # Get all patterns
    patterns = pr.get_all_patterns()
    
    # Count patterns found
    print("Candlestick Patterns Found:")
    candlestick_patterns = [
        'DOJI', 'HAMMER', 'SHOOTING_STAR', 'BULLISH_ENGULFING',
        'BEARISH_ENGULFING', 'MORNING_STAR', 'EVENING_STAR'
    ]
    
    for pattern in candlestick_patterns:
        count = patterns[pattern].sum()
        if count > 0:
            print(f"  {pattern:20} : {count:2d} occurrences")
    
    # Trend analysis
    print("\nTrend Analysis:")
    trend = pr.detect_trend(20)
    current_trend = trend.iloc[-1]
    
    if current_trend == 1:
        print("  Current Trend: 📈 UPTREND")
    elif current_trend == -1:
        print("  Current Trend: 📉 DOWNTREND")
    else:
        print("  Current Trend: ➡️  SIDEWAYS")
    
    # Support and Resistance
    print("\nSupport and Resistance Levels:")
    levels = pr.find_support_resistance(window=10, num_levels=3)
    
    print("  Support Levels:")
    for level in levels['support']:
        print(f"    ${level:.2f}")
    
    print("  Resistance Levels:")
    for level in levels['resistance']:
        print(f"    ${level:.2f}")
    
    # Chart patterns
    print("\nChart Patterns:")
    if patterns['HEAD_SHOULDERS'].any():
        print("  ⚠️  Head and Shoulders detected (bearish reversal)")
    if patterns['DOUBLE_TOP'].any():
        print("  ⚠️  Double Top detected (bearish reversal)")
    if patterns['DOUBLE_BOTTOM'].any():
        print("  🟢 Double Bottom detected (bullish reversal)")
    
    # Recent patterns
    print("\nRecent Patterns (Last 5 days):")
    recent = patterns.tail(5)
    for date, row in recent.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        found_patterns = [col for col in candlestick_patterns if row[col]]
        if found_patterns:
            print(f"  {date_str}: {', '.join(found_patterns)}")
    
    fetcher.close()


def demo_visualizations():
    """Demonstrate chart creation"""
    print_header("Interactive Charts Demo")
    
    # Fetch data
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Creating interactive charts for {symbol}...\n")
    print("Charts will be saved as HTML files in the current directory.\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )
    
    if df.empty:
        print("No data available")
        return
    
    # Initialize visualizer and technical analysis
    visualizer = StockVisualizer(df, symbol)
    ta = TechnicalAnalysis(df)
    
    # 1. Candlestick chart with moving averages
    print("1. Creating candlestick chart with moving averages...")
    sma_20 = ta.sma(20)
    sma_50 = ta.sma(50)
    
    fig1 = visualizer.candlestick_chart(
        title=f'{symbol} Price Chart with Moving Averages',
        show_volume=True,
        indicators={'SMA(20)': sma_20, 'SMA(50)': sma_50}
    )
    fig1.write_html('chart_candlestick.html')
    print("   ✓ Saved as chart_candlestick.html")
    
    # 2. Bollinger Bands
    print("2. Creating Bollinger Bands chart...")
    upper, middle, lower = ta.bollinger_bands(20, 2.0)
    fig2 = visualizer.bollinger_bands_chart(upper, middle, lower)
    fig2.write_html('chart_bollinger.html')
    print("   ✓ Saved as chart_bollinger.html")
    
    # 3. RSI
    print("3. Creating RSI indicator chart...")
    rsi = ta.rsi(14)
    fig3 = visualizer.rsi_chart(rsi)
    fig3.write_html('chart_rsi.html')
    print("   ✓ Saved as chart_rsi.html")
    
    # 4. MACD
    print("4. Creating MACD indicator chart...")
    macd_line, signal_line, histogram = ta.macd()
    fig4 = visualizer.macd_chart(macd_line, signal_line, histogram)
    fig4.write_html('chart_macd.html')
    print("   ✓ Saved as chart_macd.html")
    
    # 5. Multi-indicator dashboard
    print("5. Creating comprehensive dashboard...")
    indicators = {
        'RSI(14)': rsi,
        'MACD': macd_line
    }
    fig5 = visualizer.multi_indicator_chart(indicators)
    fig5.write_html('chart_dashboard.html')
    print("   ✓ Saved as chart_dashboard.html")
    
    # 6. Summary table
    print("6. Creating summary statistics table...")
    fig6 = create_summary_table(df, symbol)
    fig6.write_html('chart_summary.html')
    print("   ✓ Saved as chart_summary.html")
    
    print("\n✅ All charts created successfully!")
    print("   Open the HTML files in your browser to view interactive charts.")
    
    fetcher.close()


def demo_comparison():
    """Demonstrate stock comparison"""
    print_header("Stock Comparison Demo")
    
    # Fetch data for multiple stocks
    fetcher = StockDataFetcher()
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    print(f"Comparing stocks: {', '.join(symbols)}\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Fetch data for each symbol
    dfs = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        if not df.empty:
            dfs[symbol] = df
    
    if len(dfs) < 2:
        print("Not enough data for comparison")
        return
    
    # Use first stock as base
    base_symbol = symbols[0]
    base_df = dfs.pop(base_symbol)
    
    # Create comparison chart
    print("\nCreating comparison chart...")
    visualizer = StockVisualizer(base_df, base_symbol)
    
    fig = visualizer.comparison_chart(
        other_dfs=dfs,
        normalize=True,
        title='Stock Performance Comparison (Normalized)'
    )
    fig.write_html('chart_comparison.html')
    print("✓ Saved as chart_comparison.html")
    
    # Create correlation heatmap
    print("Creating correlation heatmap...")
    fig_corr = visualizer.heatmap_correlation(dfs)
    fig_corr.write_html('chart_correlation.html')
    print("✓ Saved as chart_correlation.html")
    
    fetcher.close()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  SECTION 2: DATA ANALYSIS AND VISUALIZATION DEMO")
    print("  Stock Market Analysis Platform")
    print("=" * 70)
    
    try:
        # Run demos
        demo_technical_indicators()
        demo_pattern_recognition()
        demo_visualizations()
        demo_comparison()
        
        print("\n" + "=" * 70)
        print("  Demo completed successfully!")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
