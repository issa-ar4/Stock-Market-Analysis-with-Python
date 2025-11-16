"""
Data Preparation Module for Machine Learning
Handles feature engineering, data preprocessing, and train/test splitting
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from utils.logger import logger
from data_analysis import TechnicalAnalysis


class DataPreparation:
    """
    Data preparation class for ML models
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'close'):
        """
        Initialize data preparation
        
        Args:
            df: DataFrame with OHLCV data
            target_column: Column to predict
        """
        self.df = df.copy()
        self.target_column = target_column
        self.scaler = None
        self.feature_columns = None
        logger.info(f"DataPreparation initialized with {len(df)} samples")
    
    def add_technical_features(self) -> pd.DataFrame:
        """
        Add technical indicators as features
        
        Returns:
            DataFrame with technical features
        """
        logger.info("Adding technical indicator features")
        
        ta = TechnicalAnalysis(self.df)
        
        # Add technical indicators
        result_df = self.df.copy()
        
        # Moving averages
        result_df['sma_5'] = ta.sma(5)
        result_df['sma_10'] = ta.sma(10)
        result_df['sma_20'] = ta.sma(20)
        result_df['sma_50'] = ta.sma(50)
        result_df['ema_12'] = ta.ema(12)
        result_df['ema_26'] = ta.ema(26)
        
        # Momentum indicators
        result_df['rsi'] = ta.rsi(14)
        k, d = ta.stochastic_oscillator(14, 3)
        result_df['stoch_k'] = k
        result_df['stoch_d'] = d
        result_df['momentum'] = ta.momentum(10)
        result_df['roc'] = ta.roc(12)
        
        # Volatility indicators
        upper, middle, lower = ta.bollinger_bands(20, 2.0)
        result_df['bb_upper'] = upper
        result_df['bb_middle'] = middle
        result_df['bb_lower'] = lower
        result_df['bb_width'] = (upper - lower) / middle
        result_df['atr'] = ta.atr(14)
        
        # Trend indicators
        macd_line, signal_line, histogram = ta.macd(12, 26, 9)
        result_df['macd'] = macd_line
        result_df['macd_signal'] = signal_line
        result_df['macd_hist'] = histogram
        result_df['adx'] = ta.adx(14)
        
        # Volume indicators
        result_df['obv'] = ta.obv()
        result_df['vwap'] = ta.vwap()
        result_df['mfi'] = ta.mfi(14)
        
        logger.info(f"Added {len(result_df.columns) - len(self.df.columns)} technical features")
        return result_df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with price features
        """
        logger.info("Adding price-based features")
        
        result_df = df.copy()
        
        # Price changes
        result_df['price_change'] = result_df['close'].pct_change()
        result_df['price_change_2d'] = result_df['close'].pct_change(periods=2)
        result_df['price_change_5d'] = result_df['close'].pct_change(periods=5)
        
        # High-Low range
        result_df['hl_range'] = result_df['high'] - result_df['low']
        result_df['hl_pct'] = (result_df['high'] - result_df['low']) / result_df['close']
        
        # Open-Close relationship
        result_df['oc_range'] = result_df['close'] - result_df['open']
        result_df['oc_pct'] = (result_df['close'] - result_df['open']) / result_df['open']
        
        # Gap detection
        result_df['gap'] = result_df['open'] - result_df['close'].shift(1)
        result_df['gap_pct'] = result_df['gap'] / result_df['close'].shift(1)
        
        # Volume changes
        result_df['volume_change'] = result_df['volume'].pct_change()
        result_df['volume_ma_ratio'] = result_df['volume'] / result_df['volume'].rolling(20).mean()
        
        # Price position indicators
        result_df['price_position'] = (result_df['close'] - result_df['low'].rolling(20).min()) / \
                                      (result_df['high'].rolling(20).max() - result_df['low'].rolling(20).min())
        
        logger.info(f"Added {len(result_df.columns) - len(df.columns)} price features")
        return result_df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        logger.info("Adding time-based features")
        
        result_df = df.copy()
        
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df['day_of_week'] = result_df.index.dayofweek
            result_df['day_of_month'] = result_df.index.day
            result_df['month'] = result_df.index.month
            result_df['quarter'] = result_df.index.quarter
            result_df['year'] = result_df.index.year
            result_df['week_of_year'] = result_df.index.isocalendar().week
            
            # Cyclical encoding for day of week
            result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
            
            # Cyclical encoding for month
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
            
            logger.info("Added time-based features")
        else:
            logger.warning("Index is not DatetimeIndex, skipping time features")
        
        return result_df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged features
        
        Args:
            df: DataFrame
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Adding lag features: {lags}")
        
        result_df = df.copy()
        
        for lag in lags:
            result_df[f'close_lag_{lag}'] = result_df['close'].shift(lag)
            result_df[f'volume_lag_{lag}'] = result_df['volume'].shift(lag)
            result_df[f'high_lag_{lag}'] = result_df['high'].shift(lag)
            result_df[f'low_lag_{lag}'] = result_df['low'].shift(lag)
        
        logger.info(f"Added {len(lags) * 4} lag features")
        return result_df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: DataFrame
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling features: {windows}")
        
        result_df = df.copy()
        
        for window in windows:
            # Rolling statistics for close price
            result_df[f'close_rolling_mean_{window}'] = result_df['close'].rolling(window).mean()
            result_df[f'close_rolling_std_{window}'] = result_df['close'].rolling(window).std()
            result_df[f'close_rolling_min_{window}'] = result_df['close'].rolling(window).min()
            result_df[f'close_rolling_max_{window}'] = result_df['close'].rolling(window).max()
            
            # Rolling statistics for volume
            result_df[f'volume_rolling_mean_{window}'] = result_df['volume'].rolling(window).mean()
            result_df[f'volume_rolling_std_{window}'] = result_df['volume'].rolling(window).std()
        
        logger.info(f"Added {len(windows) * 6} rolling features")
        return result_df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1, 
                     target_type: str = 'price') -> pd.DataFrame:
        """
        Create target variable for prediction
        
        Args:
            df: DataFrame
            horizon: Number of periods ahead to predict
            target_type: 'price' for actual price, 'return' for returns, 'direction' for up/down
            
        Returns:
            DataFrame with target variable
        """
        logger.info(f"Creating target variable (type={target_type}, horizon={horizon})")
        
        result_df = df.copy()
        
        if target_type == 'price':
            result_df['target'] = result_df[self.target_column].shift(-horizon)
            
        elif target_type == 'return':
            result_df['target'] = result_df[self.target_column].pct_change(periods=horizon).shift(-horizon)
            
        elif target_type == 'direction':
            future_price = result_df[self.target_column].shift(-horizon)
            result_df['target'] = (future_price > result_df[self.target_column]).astype(int)
        
        logger.info(f"Created target variable: {result_df['target'].notna().sum()} samples")
        return result_df
    
    def prepare_features(self, add_technical: bool = True, add_price: bool = True,
                        add_time: bool = True, add_lags: bool = True,
                        add_rolling: bool = True) -> pd.DataFrame:
        """
        Prepare all features
        
        Args:
            add_technical: Whether to add technical indicators
            add_price: Whether to add price features
            add_time: Whether to add time features
            add_lags: Whether to add lag features
            add_rolling: Whether to add rolling features
            
        Returns:
            DataFrame with all features
        """
        logger.info("Preparing all features")
        
        result_df = self.df.copy()
        
        if add_technical:
            result_df = self.add_technical_features()
        
        if add_price:
            result_df = self.add_price_features(result_df)
        
        if add_time:
            result_df = self.add_time_features(result_df)
        
        if add_lags:
            result_df = self.add_lag_features(result_df)
        
        if add_rolling:
            result_df = self.add_rolling_features(result_df)
        
        logger.info(f"Total features: {len(result_df.columns)}")
        return result_df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'minmax',
                      feature_columns: List[str] = None) -> Tuple[pd.DataFrame, object]:
        """
        Scale features
        
        Args:
            df: DataFrame to scale
            method: 'minmax' or 'standard'
            feature_columns: List of columns to scale (None = all numeric)
            
        Returns:
            Tuple of (scaled DataFrame, scaler object)
        """
        logger.info(f"Scaling features using {method} method")
        
        result_df = df.copy()
        
        if feature_columns is None:
            # Select all numeric columns except target
            feature_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in feature_columns:
                feature_columns.remove('target')
        
        # Initialize scaler
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        result_df[feature_columns] = scaler.fit_transform(result_df[feature_columns])
        
        self.scaler = scaler
        self.feature_columns = feature_columns
        
        logger.info(f"Scaled {len(feature_columns)} features")
        return result_df, scaler
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60,
                        feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models
        
        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            feature_columns: List of feature columns
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        logger.info(f"Creating sequences (length={sequence_length})")
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in feature_columns:
                feature_columns.remove('target')
        
        # Remove NaN values
        df_clean = df.dropna()
        
        X, y = [], []
        
        for i in range(sequence_length, len(df_clean)):
            X.append(df_clean[feature_columns].iloc[i-sequence_length:i].values)
            y.append(df_clean['target'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                   validation_size: float = 0.1, shuffle: bool = False) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data (test={test_size}, val={validation_size}, shuffle={shuffle})")
        
        # Remove NaN values
        df_clean = df.dropna()
        
        # First split: train+val and test
        train_val_size = 1 - test_size
        split_idx = int(len(df_clean) * train_val_size)
        
        if shuffle:
            df_shuffled = df_clean.sample(frac=1, random_state=42)
            train_val_df = df_shuffled.iloc[:split_idx]
            test_df = df_shuffled.iloc[split_idx:]
        else:
            train_val_df = df_clean.iloc[:split_idx]
            test_df = df_clean.iloc[split_idx:]
        
        # Second split: train and validation
        val_split_idx = int(len(train_val_df) * (1 - validation_size))
        train_df = train_val_df.iloc[:val_split_idx]
        val_df = train_val_df.iloc[val_split_idx:]
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
