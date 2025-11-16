"""
Trading Strategies for Automated Trading
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from data_analysis.technical_indicators import TechnicalAnalysis


class TradingStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.position = 0  # Current position: 0=flat, 1=long, -1=short
        self.trades = []  # List of executed trades
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals column (1=buy, -1=sell, 0=hold)
        """
        pass
    
    def get_current_signal(self, df: pd.DataFrame) -> int:
        """Get the current trading signal"""
        signals = self.generate_signals(df)
        if len(signals) > 0:
            return signals['signal'].iloc[-1]
        return 0
    
    def record_trade(self, timestamp: pd.Timestamp, signal: int, price: float, quantity: int):
        """Record a trade"""
        self.trades.append({
            'timestamp': timestamp,
            'signal': signal,
            'price': price,
            'quantity': quantity,
            'position': signal
        })


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based trading strategy
    Uses RSI and MACD indicators
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        """
        Initialize momentum strategy
        
        Args:
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
        """
        super().__init__("Momentum Strategy")
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum-based signals
        
        Buy when:
        - RSI < oversold and MACD crosses above signal
        
        Sell when:
        - RSI > overbought or MACD crosses below signal
        """
        df = df.copy()
        
        # Calculate technical indicators
        ta = TechnicalAnalysis(df)
        rsi_series = ta.rsi(period=self.rsi_period)
        macd_result = ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        # Add indicators to dataframe
        df['rsi'] = rsi_series
        if isinstance(macd_result, tuple) and len(macd_result) == 3:
            df['macd'], df['macd_signal'], df['macd_hist'] = macd_result
        elif isinstance(macd_result, pd.DataFrame):
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy signals
        buy_condition = (
            (df['rsi'] < self.rsi_oversold) &
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals
        sell_condition = (
            (df['rsi'] > self.rsi_overbought) |
            ((df['macd'] < df['macd_signal']) &
             (df['macd'].shift(1) >= df['macd_signal'].shift(1)))
        )
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion trading strategy
    Uses Bollinger Bands
    """
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_threshold: int = 50
    ):
        """
        Initialize mean reversion strategy
        
        Args:
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            rsi_period: RSI period
            rsi_threshold: RSI threshold for confirmation
        """
        super().__init__("Mean Reversion Strategy")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals
        
        Buy when:
        - Price touches or crosses below lower Bollinger Band
        - RSI < threshold (oversold confirmation)
        
        Sell when:
        - Price touches or crosses above upper Bollinger Band
        - RSI > threshold (overbought confirmation)
        """
        df = df.copy()
        
        # Calculate technical indicators
        ta = TechnicalAnalysis(df)
        bb_result = ta.bollinger_bands(period=self.bb_period, std_dev=self.bb_std)
        rsi_series = ta.rsi(period=self.rsi_period)
        
        # Add indicators to dataframe
        df['rsi'] = rsi_series
        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = bb_result
        elif isinstance(bb_result, pd.DataFrame):
            df['bb_upper'] = bb_result['bb_upper']
            df['bb_middle'] = bb_result['bb_middle']
            df['bb_lower'] = bb_result['bb_lower']
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy signals (price at lower band + oversold RSI)
        buy_condition = (
            (df['close'] <= df['bb_lower']) &
            (df['rsi'] < self.rsi_threshold)
        )
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals (price at upper band or overbought RSI)
        sell_condition = (
            (df['close'] >= df['bb_upper']) |
            (df['rsi'] > (100 - self.rsi_threshold))
        )
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class MLStrategy(TradingStrategy):
    """
    Machine Learning-based trading strategy
    Uses ensemble predictions from ML models
    """
    
    def __init__(
        self,
        model_dict: Dict,
        confidence_threshold: float = 0.6,
        lookback_period: int = 60
    ):
        """
        Initialize ML strategy
        
        Args:
            model_dict: Dictionary of trained ML models
            confidence_threshold: Minimum confidence for trading
            lookback_period: Number of days to look back for features
        """
        super().__init__("ML Strategy")
        self.models = model_dict
        self.confidence_threshold = confidence_threshold
        self.lookback_period = lookback_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-based signals
        
        Uses ensemble of models to predict price direction
        """
        from ml_models.data_preparation import DataPreparation
        
        df = df.copy()
        
        # Prepare features
        dp = DataPreparation(df)
        X, _ = dp.prepare_features(target_days=5)
        
        if len(X) == 0:
            df['signal'] = 0
            return df
        
        # Get predictions from all models
        predictions = []
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception:
                    continue
        
        if len(predictions) == 0:
            df['signal'] = 0
            return df
        
        # Ensemble: average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Align predictions with dataframe
        df['signal'] = 0
        df['ml_prediction'] = np.nan
        
        # Only keep the last len(ensemble_pred) rows
        if len(ensemble_pred) <= len(df):
            df.loc[df.index[-len(ensemble_pred):], 'ml_prediction'] = ensemble_pred
        
        # Generate signals based on prediction confidence
        df.loc[df['ml_prediction'] > self.confidence_threshold, 'signal'] = 1
        df.loc[df['ml_prediction'] < -self.confidence_threshold, 'signal'] = -1
        
        return df


class BreakoutStrategy(TradingStrategy):
    """
    Breakout trading strategy
    Trades on price breaking support/resistance levels
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        volume_threshold: float = 1.5
    ):
        """
        Initialize breakout strategy
        
        Args:
            lookback_period: Period for identifying support/resistance
            volume_threshold: Volume multiplier for confirmation
        """
        super().__init__("Breakout Strategy")
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakout signals
        
        Buy when:
        - Price breaks above resistance with high volume
        
        Sell when:
        - Price breaks below support with high volume
        """
        df = df.copy()
        
        # Calculate rolling high/low (resistance/support)
        df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
        df['support'] = df['low'].rolling(window=self.lookback_period).min()
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=self.lookback_period).mean()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy signals (breakout above resistance)
        buy_condition = (
            (df['close'] > df['resistance'].shift(1)) &
            (df['volume'] > df['avg_volume'] * self.volume_threshold)
        )
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals (breakdown below support)
        sell_condition = (
            (df['close'] < df['support'].shift(1)) &
            (df['volume'] > df['avg_volume'] * self.volume_threshold)
        )
        df.loc[sell_condition, 'signal'] = -1
        
        return df
