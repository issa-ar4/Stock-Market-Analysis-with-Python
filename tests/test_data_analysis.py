"""
Unit tests for data analysis module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_analysis import TechnicalAnalysis, PatternRecognition


class TestTechnicalAnalysis:
    """Test technical indicators"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, 100)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        self.ta = TechnicalAnalysis(self.df)
    
    def test_sma(self):
        """Test Simple Moving Average"""
        sma = self.ta.sma(20)
        
        assert len(sma) == len(self.df)
        assert not sma.iloc[-1] is None
        assert sma.iloc[-1] > 0
    
    def test_ema(self):
        """Test Exponential Moving Average"""
        ema = self.ta.ema(20)
        
        assert len(ema) == len(self.df)
        assert not ema.iloc[-1] is None
    
    def test_rsi(self):
        """Test RSI calculation"""
        rsi = self.ta.rsi(14)
        
        assert len(rsi) == len(self.df)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = self.ta.macd()
        
        assert len(macd_line) == len(self.df)
        assert len(signal_line) == len(self.df)
        assert len(histogram) == len(self.df)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands"""
        upper, middle, lower = self.ta.bollinger_bands(20, 2.0)
        
        assert len(upper) == len(self.df)
        assert len(middle) == len(self.df)
        assert len(lower) == len(self.df)
        
        # Upper should be > middle > lower
        valid_idx = ~upper.isna()
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()
    
    def test_atr(self):
        """Test Average True Range"""
        atr = self.ta.atr(14)
        
        assert len(atr) == len(self.df)
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator"""
        k, d = self.ta.stochastic_oscillator(14, 3)
        
        assert len(k) == len(self.df)
        assert len(d) == len(self.df)
        
        # Values should be between 0 and 100
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
    
    def test_obv(self):
        """Test On-Balance Volume"""
        obv = self.ta.obv()
        
        assert len(obv) == len(self.df)
        assert not obv.iloc[0] is None
    
    def test_add_all_indicators(self):
        """Test adding all indicators"""
        result = self.ta.add_all_indicators()
        
        assert len(result) == len(self.df)
        assert 'SMA_20' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_UPPER' in result.columns
    
    def test_get_trading_signals(self):
        """Test trading signal generation"""
        signals = self.ta.get_trading_signals()
        
        assert len(signals) == len(self.df)
        assert 'RSI_OVERSOLD' in signals.columns
        assert 'MACD_BULLISH' in signals.columns


class TestPatternRecognition:
    """Test pattern recognition"""
    
    def setup_method(self):
        """Setup test fixtures"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, 100)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        self.pr = PatternRecognition(self.df)
    
    def test_doji(self):
        """Test Doji pattern detection"""
        doji = self.pr.doji()
        
        assert len(doji) == len(self.df)
        assert isinstance(doji, pd.Series)
        assert doji.dtype == bool
    
    def test_hammer(self):
        """Test Hammer pattern detection"""
        hammer = self.pr.hammer()
        
        assert len(hammer) == len(self.df)
        assert isinstance(hammer, pd.Series)
    
    def test_engulfing_patterns(self):
        """Test Engulfing patterns"""
        bullish = self.pr.engulfing_bullish()
        bearish = self.pr.engulfing_bearish()
        
        assert len(bullish) == len(self.df)
        assert len(bearish) == len(self.df)
    
    def test_detect_trend(self):
        """Test trend detection"""
        trend = self.pr.detect_trend(20)
        
        assert len(trend) == len(self.df)
        # Trend should be -1, 0, or 1
        valid_trend = trend.dropna()
        assert set(valid_trend.unique()).issubset({-1, 0, 1})
    
    def test_support_resistance(self):
        """Test support and resistance levels"""
        levels = self.pr.find_support_resistance(window=10, num_levels=3)
        
        assert 'support' in levels
        assert 'resistance' in levels
        assert isinstance(levels['support'], list)
        assert isinstance(levels['resistance'], list)
    
    def test_get_all_patterns(self):
        """Test getting all patterns"""
        patterns = self.pr.get_all_patterns()
        
        assert len(patterns) == len(self.df)
        assert 'DOJI' in patterns.columns
        assert 'HAMMER' in patterns.columns
        assert 'TREND' in patterns.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
