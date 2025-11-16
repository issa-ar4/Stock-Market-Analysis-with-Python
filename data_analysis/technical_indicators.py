"""
Technical Indicators Module
Implements various technical analysis indicators for stock data
"""
import pandas as pd
import numpy as np
from typing import Union, Tuple
from utils.logger import logger


class TechnicalAnalysis:
    """
    Technical Analysis class for calculating various technical indicators
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self._validate_dataframe()
        logger.info(f"TechnicalAnalysis initialized with {len(df)} data points")
    
    def _validate_dataframe(self):
        """Validate that DataFrame has required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    # ==================== Moving Averages ====================
    
    def sma(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            period: Number of periods
            column: Column to calculate SMA on
            
        Returns:
            Series with SMA values
        """
        logger.info(f"Calculating SMA(period={period})")
        return self.df[column].rolling(window=period).mean()
    
    def ema(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            period: Number of periods
            column: Column to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        logger.info(f"Calculating EMA(period={period})")
        return self.df[column].ewm(span=period, adjust=False).mean()
    
    def wma(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Weighted Moving Average
        
        Args:
            period: Number of periods
            column: Column to calculate WMA on
            
        Returns:
            Series with WMA values
        """
        logger.info(f"Calculating WMA(period={period})")
        weights = np.arange(1, period + 1)
        
        def weighted_average(values):
            return np.dot(values, weights) / weights.sum()
        
        return self.df[column].rolling(window=period).apply(weighted_average, raw=True)
    
    # ==================== Momentum Indicators ====================
    
    def rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            period: Number of periods (typically 14)
            column: Column to calculate RSI on
            
        Returns:
            Series with RSI values (0-100)
        """
        logger.info(f"Calculating RSI(period={period})")
        
        # Calculate price changes
        delta = self.df[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic_oscillator(self, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator (%K and %D)
        
        Args:
            k_period: Period for %K line
            d_period: Period for %D line (SMA of %K)
            
        Returns:
            Tuple of (%K, %D) Series
        """
        logger.info(f"Calculating Stochastic Oscillator(K={k_period}, D={d_period})")
        
        # Calculate %K
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        k = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def momentum(self, period: int = 10, column: str = 'close') -> pd.Series:
        """
        Momentum Indicator
        
        Args:
            period: Number of periods
            column: Column to calculate momentum on
            
        Returns:
            Series with momentum values
        """
        logger.info(f"Calculating Momentum(period={period})")
        return self.df[column].diff(period)
    
    def roc(self, period: int = 12, column: str = 'close') -> pd.Series:
        """
        Rate of Change (ROC)
        
        Args:
            period: Number of periods
            column: Column to calculate ROC on
            
        Returns:
            Series with ROC percentage values
        """
        logger.info(f"Calculating ROC(period={period})")
        return ((self.df[column] - self.df[column].shift(period)) / 
                self.df[column].shift(period)) * 100
    
    # ==================== Volatility Indicators ====================
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0, 
                       column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            period: Number of periods for SMA
            std_dev: Number of standard deviations
            column: Column to calculate bands on
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        logger.info(f"Calculating Bollinger Bands(period={period}, std={std_dev})")
        
        middle_band = self.df[column].rolling(window=period).mean()
        std = self.df[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Args:
            period: Number of periods
            
        Returns:
            Series with ATR values
        """
        logger.info(f"Calculating ATR(period={period})")
        
        # Calculate True Range
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def standard_deviation(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Standard Deviation (volatility measure)
        
        Args:
            period: Number of periods
            column: Column to calculate on
            
        Returns:
            Series with standard deviation values
        """
        logger.info(f"Calculating Standard Deviation(period={period})")
        return self.df[column].rolling(window=period).std()
    
    # ==================== Trend Indicators ====================
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, 
            column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        logger.info(f"Calculating MACD(fast={fast}, slow={slow}, signal={signal})")
        
        # Calculate EMAs
        ema_fast = self.df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df[column].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def adx(self, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX) - Trend Strength
        
        Args:
            period: Number of periods
            
        Returns:
            Series with ADX values
        """
        logger.info(f"Calculating ADX(period={period})")
        
        # Calculate +DM and -DM
        high_diff = self.df['high'].diff()
        low_diff = -self.df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self.atr(period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def cci(self, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        
        Args:
            period: Number of periods
            
        Returns:
            Series with CCI values
        """
        logger.info(f"Calculating CCI(period={period})")
        
        # Calculate Typical Price
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Calculate SMA of Typical Price
        sma_tp = tp.rolling(window=period).mean()
        
        # Calculate Mean Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Calculate CCI
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return cci
    
    # ==================== Volume Indicators ====================
    
    def obv(self) -> pd.Series:
        """
        On-Balance Volume (OBV)
        
        Returns:
            Series with OBV values
        """
        logger.info("Calculating OBV")
        
        obv = pd.Series(index=self.df.index, dtype=float)
        obv.iloc[0] = self.df['volume'].iloc[0]
        
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + self.df['volume'].iloc[i]
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - self.df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        
        return obv
    
    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Returns:
            Series with VWAP values
        """
        logger.info("Calculating VWAP")
        
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (typical_price * self.df['volume']).cumsum() / self.df['volume'].cumsum()
        
        return vwap
    
    def mfi(self, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI)
        
        Args:
            period: Number of periods
            
        Returns:
            Series with MFI values (0-100)
        """
        logger.info(f"Calculating MFI(period={period})")
        
        # Calculate Typical Price
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Calculate Money Flow
        money_flow = tp * self.df['volume']
        
        # Separate positive and negative money flow
        positive_flow = money_flow.where(tp > tp.shift(1), 0)
        negative_flow = money_flow.where(tp < tp.shift(1), 0)
        
        # Calculate Money Flow Ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfr = positive_mf / negative_mf
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    # ==================== Helper Methods ====================
    
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame
        
        Returns:
            DataFrame with all indicators added
        """
        logger.info("Adding all technical indicators")
        
        result_df = self.df.copy()
        
        # Moving Averages
        result_df['SMA_20'] = self.sma(20)
        result_df['SMA_50'] = self.sma(50)
        result_df['SMA_200'] = self.sma(200)
        result_df['EMA_12'] = self.ema(12)
        result_df['EMA_26'] = self.ema(26)
        
        # Momentum
        result_df['RSI'] = self.rsi(14)
        k, d = self.stochastic_oscillator()
        result_df['STOCH_K'] = k
        result_df['STOCH_D'] = d
        
        # Volatility
        upper, middle, lower = self.bollinger_bands()
        result_df['BB_UPPER'] = upper
        result_df['BB_MIDDLE'] = middle
        result_df['BB_LOWER'] = lower
        result_df['ATR'] = self.atr(14)
        
        # Trend
        macd_line, signal_line, histogram = self.macd()
        result_df['MACD'] = macd_line
        result_df['MACD_SIGNAL'] = signal_line
        result_df['MACD_HIST'] = histogram
        result_df['ADX'] = self.adx(14)
        
        # Volume
        result_df['OBV'] = self.obv()
        result_df['VWAP'] = self.vwap()
        result_df['MFI'] = self.mfi(14)
        
        logger.info(f"Added {len(result_df.columns) - len(self.df.columns)} indicators")
        return result_df
    
    def get_trading_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Returns:
            DataFrame with trading signals
        """
        logger.info("Generating trading signals")
        
        signals = pd.DataFrame(index=self.df.index)
        
        # RSI signals
        rsi = self.rsi(14)
        signals['RSI_OVERSOLD'] = rsi < 30
        signals['RSI_OVERBOUGHT'] = rsi > 70
        
        # MACD signals
        macd_line, signal_line, _ = self.macd()
        signals['MACD_BULLISH'] = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals['MACD_BEARISH'] = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Moving Average Crossover
        sma_20 = self.sma(20)
        sma_50 = self.sma(50)
        signals['MA_GOLDEN_CROSS'] = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
        signals['MA_DEATH_CROSS'] = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
        
        # Bollinger Bands signals
        upper, middle, lower = self.bollinger_bands()
        signals['BB_LOWER_TOUCH'] = self.df['close'] <= lower
        signals['BB_UPPER_TOUCH'] = self.df['close'] >= upper
        
        return signals
