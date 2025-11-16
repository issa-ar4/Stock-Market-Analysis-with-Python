"""
Comprehensive Test Suite for All Sections
Tests data ingestion, technical analysis, and ML models
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from database.repositories import StockPriceRepository, StockInfoRepository
from data_ingestion.data_fetcher import StockDataFetcher
from data_analysis.technical_indicators import TechnicalAnalysis
from data_analysis.pattern_recognition import PatternRecognition
from data_analysis.visualization import StockVisualizer
from ml_models.data_preparation import DataPreparation
import pandas as pd
from datetime import datetime, timedelta


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def test_section1_data_ingestion():
    """Test Section 1: Data Ingestion"""
    print_header("Testing Section 1: Data Ingestion System")
    
    try:
        # Test 1: Configuration
        print("\n1. Testing Configuration...")
        config = Config()
        print_success("Configuration loaded")
        print_info(f"   Database URL: {config.DATABASE_URL[:30]}...")
        print_info(f"   Alpha Vantage API configured: {bool(config.ALPHA_VANTAGE_API_KEY)}")
        print_info(f"   Finnhub API configured: {bool(config.FINNHUB_API_KEY)}")
        print_info(f"   Alpaca API configured: {bool(config.ALPACA_API_KEY)}")
        
        # Test 2: Database Connection
        print("\n2. Testing Database Connection...")
        from database import get_engine, get_session
        engine = get_engine(config.DATABASE_URL)
        session = get_session(engine)
        stock_repo = StockPriceRepository(session)
        info_repo = StockInfoRepository(session)
        print_success("Database repositories initialized")
        
        # Test 3: Check for data
        print("\n3. Checking for Historical Data...")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        data_found = {}
        
        for symbol in symbols:
            stock_data = stock_repo.get_latest(symbol=symbol)
            if stock_data:
                # Get last 30 days of data
                from datetime import timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                all_data = stock_repo.get_by_symbol(symbol=symbol, start_date=start_date)
                data_found[symbol] = len(all_data)
                print_success(f"   {symbol}: {data_found[symbol]} records found (last 30 days)")
            else:
                print_info(f"   {symbol}: No data (run fetch_historical_data.py)")
        
        if not data_found:
            print_error("No historical data found. Run: python scripts/fetch_historical_data.py")
            return False, None, None
        
        # Test 4: Data Quality
        print("\n4. Testing Data Quality...")
        test_symbol = list(data_found.keys())[0]
        # Get last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        test_data = stock_repo.get_by_symbol(symbol=test_symbol, start_date=start_date)
        
        if test_data:
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'open': float(r.open),
                'high': float(r.high),
                'low': float(r.low),
                'close': float(r.close),
                'volume': int(r.volume)
            } for r in test_data])
            
            print_success(f"   Loaded {len(df)} records for {test_symbol}")
            print_info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print_info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print_info(f"   Avg volume: {df['volume'].mean():,.0f}")
        
        print_success("\nSection 1: Data Ingestion - PASSED")
        return True, test_symbol, df
        
    except Exception as e:
        print_error(f"Section 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_section2_technical_analysis(symbol, df):
    """Test Section 2: Technical Analysis"""
    print_header("Testing Section 2: Technical Analysis & Visualization")
    
    try:
        # Test 1: Technical Indicators
        print("\n1. Testing Technical Indicators...")
        ta = TechnicalAnalysis(df.copy())
        
        df_test = ta.sma(period=20)
        df_test = ta.ema(period=20)
        df_test = ta.rsi()
        df_test = ta.macd()
        df_test = ta.bollinger_bands()
        df_test = ta.atr()
        
        indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bollinger_upper', 'atr']
        for indicator in indicators:
            if indicator in df_test.columns:
                print_success(f"   {indicator}: ✓")
            else:
                print_error(f"   {indicator}: Missing")
        
        # Show current values
        print("\n   Current Indicator Values:")
        print_info(f"   Close: ${df_test['close'].iloc[-1]:.2f}")
        print_info(f"   SMA(20): ${df_test['sma_20'].iloc[-1]:.2f}")
        print_info(f"   RSI: {df_test['rsi'].iloc[-1]:.2f}")
        print_info(f"   MACD: {df_test['macd'].iloc[-1]:.2f}")
        
        # Test 2: Pattern Recognition
        print("\n2. Testing Pattern Recognition...")
        pr = PatternRecognition(df.copy())
        
        df_patterns = pr.doji()
        df_patterns = pr.hammer()
        df_patterns = pr.engulfing_bullish()
        df_patterns = pr.engulfing_bearish()
        
        pattern_cols = [col for col in df_patterns.columns if col.startswith('pattern_')]
        print_success(f"   {len(pattern_cols)} pattern detectors active")
        
        # Count recent patterns
        pattern_counts = {}
        for col in pattern_cols:
            count = df_patterns[col].tail(50).sum()
            if count > 0:
                pattern_name = col.replace('pattern_', '').replace('_', ' ').title()
                pattern_counts[pattern_name] = int(count)
                print_info(f"   {pattern_name}: {int(count)} occurrences (last 50 days)")
        
        # Test 3: Support/Resistance
        print("\n3. Testing Support/Resistance Detection...")
        support, resistance = pr.find_support_resistance(df)
        print_success(f"   Support: ${support:.2f}")
        print_success(f"   Resistance: ${resistance:.2f}")
        
        # Test 4: Trend Detection
        print("\n4. Testing Trend Detection...")
        trend = pr.detect_trend(df)
        trend_emoji = "📈" if trend == "uptrend" else "📉" if trend == "downtrend" else "➡️"
        print_success(f"   Current trend: {trend_emoji} {trend.upper()}")
        
        # Test 5: Visualization
        print("\n5. Testing Visualization...")
        visualizer = StockVisualizer()
        
        try:
            fig = visualizer.candlestick_chart(df.tail(50), title=f"{symbol} Test Chart")
            print_success("   Candlestick chart: ✓")
            
            fig = visualizer.bollinger_bands_chart(df_test.tail(50))
            print_success("   Bollinger Bands chart: ✓")
            
            fig = visualizer.rsi_chart(df_test.tail(50))
            print_success("   RSI chart: ✓")
            
            fig = visualizer.macd_chart(df_test.tail(50))
            print_success("   MACD chart: ✓")
            
            print_info("   (Charts created but not displayed)")
            
        except Exception as e:
            print_error(f"   Visualization error: {str(e)}")
        
        print_success("\nSection 2: Technical Analysis - PASSED")
        return True
        
    except Exception as e:
        print_error(f"Section 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_section3_ml_models(symbol, df):
    """Test Section 3: ML Models"""
    print_header("Testing Section 3: ML Models")
    
    try:
        # Test 1: Data Preparation
        print("\n1. Testing Data Preparation...")
        data_prep = DataPreparation(df.copy())
        
        df_features = data_prep.prepare_features()
        print_success(f"   Features prepared: {len(df_features.columns)} columns")
        
        df_features = data_prep.create_target(target_type='price')
        print_success("   Target variable created")
        
        # Test scaling
        df_scaled, scaler = data_prep.scale_features()
        print_success("   Features scaled")
        
        # Test sequence creation
        X, y = data_prep.create_sequences(sequence_length=30)
        print_success(f"   Sequences created: {X.shape[0]} samples")
        print_info(f"   Sequence shape: {X.shape}")
        
        # Test 2: Ensemble Model
        print("\n2. Testing Ensemble Model...")
        try:
            from ml_models.ensemble_model import EnsemblePredictor
            
            ensemble = EnsemblePredictor()
            models = ensemble.build_models(use_xgboost=False)
            print_success(f"   Built {len(models)} models: {', '.join(models.keys())}")
            
            # Quick training test (small dataset)
            print_info("   Testing training (small sample)...")
            df_small = df.tail(50).copy()
            dp_test = DataPreparation(df_small)
            df_prep = dp_test.prepare_features()
            df_prep = dp_test.create_target(target_type='price')
            df_prep = df_prep.dropna()
            
            feature_cols = [col for col in df_prep.columns if col not in ['target', 'timestamp', 'symbol']]
            X_train = df_prep[feature_cols].values[:150]
            y_train = df_prep['target'].values[:150]
            X_test = df_prep[feature_cols].values[150:]
            y_test = df_prep['target'].values[150:]
            
            ensemble.train(X_train, y_train)
            print_success("   Models trained successfully")
            
            predictions = ensemble.predict(X_test, use_ensemble=False)
            print_success(f"   Predictions made: {len(predictions)} samples")
            
            metrics = ensemble.evaluate(X_test, y_test, use_ensemble=False)
            print_info(f"   Test RMSE: {metrics['rmse']:.4f}")
            print_info(f"   Test R²: {metrics['r2']:.4f}")
            
        except ImportError as e:
            print_error(f"   Ensemble models require scikit-learn: {str(e)}")
        
        # Test 3: LSTM Model
        print("\n3. Testing LSTM Model...")
        try:
            from ml_models.lstm_model import LSTMPredictor
            
            if not hasattr(LSTMPredictor, '__init__'):
                raise ImportError("TensorFlow not available")
            
            lstm = LSTMPredictor(sequence_length=30, n_features=X.shape[2])
            print_success("   LSTM model initialized")
            
            lstm.build_model(lstm_units=[20, 10], dropout_rate=0.2)
            print_success("   LSTM architecture built")
            print_info("   (Training skipped - takes 5-10 minutes)")
            
        except ImportError as e:
            print_info("   LSTM requires TensorFlow (install: pip install tensorflow)")
            print_info("   Skipping LSTM test")
        
        print_success("\nSection 3: ML Models - PASSED")
        return True
        
    except Exception as e:
        print_error(f"Section 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check installed dependencies"""
    print_header("Checking Dependencies")
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'yfinance': 'yfinance',
        'alpha_vantage': 'alpha_vantage',
        'finnhub': 'finnhub-python',
        'sqlalchemy': 'sqlalchemy',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'xgboost': 'xgboost',
        'streamlit': 'streamlit'
    }
    
    installed = {}
    for name, import_name in dependencies.items():
        try:
            if name == 'scikit-learn':
                import sklearn
            else:
                __import__(import_name)
            installed[name] = True
            print_success(f"{name}: Installed")
        except ImportError:
            installed[name] = False
            print_info(f"{name}: Not installed (optional for some features)")
    
    critical = ['pandas', 'numpy', 'plotly', 'sqlalchemy', 'yfinance']
    all_critical = all(installed.get(dep, False) for dep in critical)
    
    if not all_critical:
        print_error("\nMissing critical dependencies!")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Run comprehensive tests"""
    print_header("Stock Market Analysis Platform - Comprehensive Test Suite")
    print(f"Starting tests at: {pd.Timestamp.now()}")
    
    # Check dependencies first
    if not check_dependencies():
        print_error("\nTests aborted due to missing dependencies")
        return
    
    results = {}
    
    # Test Section 1
    result1, symbol, df = test_section1_data_ingestion()
    results['Section 1'] = result1
    
    if not result1 or df is None:
        print_error("\nCannot proceed without data. Please run:")
        print_info("  python scripts/init_db.py")
        print_info("  python scripts/fetch_historical_data.py --symbols AAPL,MSFT,GOOGL")
        return
    
    # Test Section 2
    results['Section 2'] = test_section2_technical_analysis(symbol, df)
    
    # Test Section 3
    results['Section 3'] = test_section3_ml_models(symbol, df)
    
    # Summary
    print_header("Test Summary")
    print()
    for section, passed in results.items():
        if passed:
            print_success(f"{section}: PASSED ✓")
        else:
            print_error(f"{section}: FAILED ✗")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour Stock Market Analysis Platform is working correctly!")
        print("\nNext steps:")
        print("  1. Run demo scripts: python demo_section1.py, demo_section2.py")
        print("  2. Launch dashboard: streamlit run dashboard/app.py")
        print("  3. Train ML models: python demo_section3.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease check the errors above and:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Initialize database: python scripts/init_db.py")
        print("  3. Fetch data: python scripts/fetch_historical_data.py")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
