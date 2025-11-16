"""
Pattern Recognition Module
Identifies chart patterns and price action patterns
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from utils.logger import logger


class PatternRecognition:
    """
    Pattern Recognition class for identifying common chart patterns
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        logger.info(f"PatternRecognition initialized with {len(df)} data points")
    
    # ==================== Candlestick Patterns ====================
    
    def doji(self, threshold: float = 0.1) -> pd.Series:
        """
        Identify Doji candlestick pattern
        
        Args:
            threshold: Max body size as % of total range
            
        Returns:
            Boolean Series indicating Doji patterns
        """
        body = abs(self.df['close'] - self.df['open'])
        range_size = self.df['high'] - self.df['low']
        
        # Avoid division by zero
        range_size = range_size.replace(0, np.nan)
        
        is_doji = (body / range_size) < threshold
        return is_doji.fillna(False)
    
    def hammer(self, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
        """
        Identify Hammer pattern (bullish reversal)
        
        Args:
            body_ratio: Max body size as ratio of total range
            shadow_ratio: Min lower shadow to body ratio
            
        Returns:
            Boolean Series indicating Hammer patterns
        """
        body = abs(self.df['close'] - self.df['open'])
        total_range = self.df['high'] - self.df['low']
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        
        # Avoid division by zero
        body = body.replace(0, 0.001)
        total_range = total_range.replace(0, np.nan)
        
        is_hammer = (
            (body / total_range < body_ratio) &
            (lower_shadow / body > shadow_ratio) &
            (upper_shadow < body)
        )
        
        return is_hammer.fillna(False)
    
    def shooting_star(self, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
        """
        Identify Shooting Star pattern (bearish reversal)
        
        Args:
            body_ratio: Max body size as ratio of total range
            shadow_ratio: Min upper shadow to body ratio
            
        Returns:
            Boolean Series indicating Shooting Star patterns
        """
        body = abs(self.df['close'] - self.df['open'])
        total_range = self.df['high'] - self.df['low']
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        
        # Avoid division by zero
        body = body.replace(0, 0.001)
        total_range = total_range.replace(0, np.nan)
        
        is_shooting_star = (
            (body / total_range < body_ratio) &
            (upper_shadow / body > shadow_ratio) &
            (lower_shadow < body)
        )
        
        return is_shooting_star.fillna(False)
    
    def engulfing_bullish(self) -> pd.Series:
        """
        Identify Bullish Engulfing pattern
        
        Returns:
            Boolean Series indicating Bullish Engulfing patterns
        """
        prev_bearish = self.df['close'].shift(1) < self.df['open'].shift(1)
        curr_bullish = self.df['close'] > self.df['open']
        
        curr_body_larger = (
            (self.df['close'] - self.df['open']) >
            (self.df['open'].shift(1) - self.df['close'].shift(1))
        )
        
        engulfs = (
            (self.df['open'] < self.df['close'].shift(1)) &
            (self.df['close'] > self.df['open'].shift(1))
        )
        
        is_bullish_engulfing = prev_bearish & curr_bullish & curr_body_larger & engulfs
        return is_bullish_engulfing.fillna(False)
    
    def engulfing_bearish(self) -> pd.Series:
        """
        Identify Bearish Engulfing pattern
        
        Returns:
            Boolean Series indicating Bearish Engulfing patterns
        """
        prev_bullish = self.df['close'].shift(1) > self.df['open'].shift(1)
        curr_bearish = self.df['close'] < self.df['open']
        
        curr_body_larger = (
            (self.df['open'] - self.df['close']) >
            (self.df['close'].shift(1) - self.df['open'].shift(1))
        )
        
        engulfs = (
            (self.df['open'] > self.df['close'].shift(1)) &
            (self.df['close'] < self.df['open'].shift(1))
        )
        
        is_bearish_engulfing = prev_bullish & curr_bearish & curr_body_larger & engulfs
        return is_bearish_engulfing.fillna(False)
    
    def morning_star(self, window: int = 3) -> pd.Series:
        """
        Identify Morning Star pattern (bullish reversal)
        
        Args:
            window: Number of candles to consider
            
        Returns:
            Boolean Series indicating Morning Star patterns
        """
        # First candle: bearish
        first_bearish = self.df['close'].shift(2) < self.df['open'].shift(2)
        
        # Second candle: small body (star)
        star_body = abs(self.df['close'].shift(1) - self.df['open'].shift(1))
        star_small = star_body < (self.df['high'].shift(1) - self.df['low'].shift(1)) * 0.3
        
        # Third candle: bullish and closes above midpoint of first candle
        third_bullish = self.df['close'] > self.df['open']
        closes_high = self.df['close'] > (self.df['open'].shift(2) + self.df['close'].shift(2)) / 2
        
        is_morning_star = first_bearish & star_small & third_bullish & closes_high
        return is_morning_star.fillna(False)
    
    def evening_star(self, window: int = 3) -> pd.Series:
        """
        Identify Evening Star pattern (bearish reversal)
        
        Args:
            window: Number of candles to consider
            
        Returns:
            Boolean Series indicating Evening Star patterns
        """
        # First candle: bullish
        first_bullish = self.df['close'].shift(2) > self.df['open'].shift(2)
        
        # Second candle: small body (star)
        star_body = abs(self.df['close'].shift(1) - self.df['open'].shift(1))
        star_small = star_body < (self.df['high'].shift(1) - self.df['low'].shift(1)) * 0.3
        
        # Third candle: bearish and closes below midpoint of first candle
        third_bearish = self.df['close'] < self.df['open']
        closes_low = self.df['close'] < (self.df['open'].shift(2) + self.df['close'].shift(2)) / 2
        
        is_evening_star = first_bullish & star_small & third_bearish & closes_low
        return is_evening_star.fillna(False)
    
    # ==================== Support and Resistance ====================
    
    def find_support_resistance(self, window: int = 20, num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels
        
        Args:
            window: Window size for finding local min/max
            num_levels: Number of levels to identify
            
        Returns:
            Dictionary with support and resistance levels
        """
        logger.info(f"Finding support/resistance levels (window={window}, num_levels={num_levels})")
        
        # Find local minima (support) and maxima (resistance)
        supports = []
        resistances = []
        
        for i in range(window, len(self.df) - window):
            # Check if current point is local minimum
            if self.df['low'].iloc[i] == self.df['low'].iloc[i-window:i+window+1].min():
                supports.append(self.df['low'].iloc[i])
            
            # Check if current point is local maximum
            if self.df['high'].iloc[i] == self.df['high'].iloc[i-window:i+window+1].max():
                resistances.append(self.df['high'].iloc[i])
        
        # Cluster nearby levels
        def cluster_levels(levels, tolerance=0.02):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        support_levels = cluster_levels(supports)[-num_levels:]
        resistance_levels = cluster_levels(resistances)[-num_levels:]
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }
    
    # ==================== Trend Detection ====================
    
    def detect_trend(self, window: int = 20) -> pd.Series:
        """
        Detect price trend direction
        
        Args:
            window: Window size for trend calculation
            
        Returns:
            Series with trend values: 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        logger.info(f"Detecting trends (window={window})")
        
        # Calculate moving average slope
        ma = self.df['close'].rolling(window=window).mean()
        slope = ma.diff(window)
        
        # Classify trend
        trend = pd.Series(0, index=self.df.index)
        trend[slope > 0] = 1  # Uptrend
        trend[slope < 0] = -1  # Downtrend
        
        return trend
    
    def higher_highs_lows(self, window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Identify higher highs and higher lows (bullish trend)
        
        Args:
            window: Window for comparison
            
        Returns:
            Tuple of (higher_highs, higher_lows) boolean Series
        """
        higher_highs = self.df['high'] > self.df['high'].shift(window)
        higher_lows = self.df['low'] > self.df['low'].shift(window)
        
        return higher_highs, higher_lows
    
    def lower_highs_lows(self, window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Identify lower highs and lower lows (bearish trend)
        
        Args:
            window: Window for comparison
            
        Returns:
            Tuple of (lower_highs, lower_lows) boolean Series
        """
        lower_highs = self.df['high'] < self.df['high'].shift(window)
        lower_lows = self.df['low'] < self.df['low'].shift(window)
        
        return lower_highs, lower_lows
    
    # ==================== Chart Patterns ====================
    
    def head_and_shoulders(self, window: int = 10, tolerance: float = 0.03) -> pd.Series:
        """
        Identify Head and Shoulders pattern (simplified)
        
        Args:
            window: Window for finding peaks
            tolerance: Tolerance for shoulder height matching
            
        Returns:
            Boolean Series indicating potential H&S patterns
        """
        logger.info("Detecting Head and Shoulders patterns")
        
        # Find local peaks
        peaks = pd.Series(False, index=self.df.index)
        
        for i in range(window, len(self.df) - window):
            if self.df['high'].iloc[i] == self.df['high'].iloc[i-window:i+window+1].max():
                peaks.iloc[i] = True
        
        # Look for three consecutive peaks where middle is highest
        pattern = pd.Series(False, index=self.df.index)
        peak_indices = peaks[peaks].index
        
        for i in range(2, len(peak_indices)):
            idx1, idx2, idx3 = peak_indices[i-2], peak_indices[i-1], peak_indices[i]
            
            h1 = self.df.loc[idx1, 'high']
            h2 = self.df.loc[idx2, 'high']
            h3 = self.df.loc[idx3, 'high']
            
            # Check if middle peak is highest and shoulders are similar
            if (h2 > h1 and h2 > h3 and 
                abs(h1 - h3) / h1 < tolerance):
                pattern.loc[idx3] = True
        
        return pattern
    
    def double_top_bottom(self, window: int = 10, tolerance: float = 0.02) -> Dict[str, pd.Series]:
        """
        Identify Double Top and Double Bottom patterns
        
        Args:
            window: Window for finding peaks/troughs
            tolerance: Tolerance for price matching
            
        Returns:
            Dictionary with 'double_top' and 'double_bottom' Series
        """
        logger.info("Detecting Double Top/Bottom patterns")
        
        # Find peaks and troughs
        peaks = pd.Series(False, index=self.df.index)
        troughs = pd.Series(False, index=self.df.index)
        
        for i in range(window, len(self.df) - window):
            if self.df['high'].iloc[i] == self.df['high'].iloc[i-window:i+window+1].max():
                peaks.iloc[i] = True
            if self.df['low'].iloc[i] == self.df['low'].iloc[i-window:i+window+1].min():
                troughs.iloc[i] = True
        
        # Detect double tops
        double_top = pd.Series(False, index=self.df.index)
        peak_indices = peaks[peaks].index
        
        for i in range(1, len(peak_indices)):
            idx1, idx2 = peak_indices[i-1], peak_indices[i]
            p1 = self.df.loc[idx1, 'high']
            p2 = self.df.loc[idx2, 'high']
            
            if abs(p1 - p2) / p1 < tolerance:
                double_top.loc[idx2] = True
        
        # Detect double bottoms
        double_bottom = pd.Series(False, index=self.df.index)
        trough_indices = troughs[troughs].index
        
        for i in range(1, len(trough_indices)):
            idx1, idx2 = trough_indices[i-1], trough_indices[i]
            t1 = self.df.loc[idx1, 'low']
            t2 = self.df.loc[idx2, 'low']
            
            if abs(t1 - t2) / t1 < tolerance:
                double_bottom.loc[idx2] = True
        
        return {
            'double_top': double_top,
            'double_bottom': double_bottom
        }
    
    # ==================== Summary ====================
    
    def get_all_patterns(self) -> pd.DataFrame:
        """
        Identify all patterns and return summary DataFrame
        
        Returns:
            DataFrame with all pattern indicators
        """
        logger.info("Identifying all patterns")
        
        patterns = pd.DataFrame(index=self.df.index)
        
        # Candlestick patterns
        patterns['DOJI'] = self.doji()
        patterns['HAMMER'] = self.hammer()
        patterns['SHOOTING_STAR'] = self.shooting_star()
        patterns['BULLISH_ENGULFING'] = self.engulfing_bullish()
        patterns['BEARISH_ENGULFING'] = self.engulfing_bearish()
        patterns['MORNING_STAR'] = self.morning_star()
        patterns['EVENING_STAR'] = self.evening_star()
        
        # Trend
        patterns['TREND'] = self.detect_trend()
        
        # Chart patterns
        patterns['HEAD_SHOULDERS'] = self.head_and_shoulders()
        double_patterns = self.double_top_bottom()
        patterns['DOUBLE_TOP'] = double_patterns['double_top']
        patterns['DOUBLE_BOTTOM'] = double_patterns['double_bottom']
        
        logger.info(f"Identified patterns in {patterns.sum().sum()} instances")
        return patterns
