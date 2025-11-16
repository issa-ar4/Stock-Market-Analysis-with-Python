"""
Model Trainer for coordinating ML model training and evaluation
Handles cross-validation, hyperparameter tuning, and model comparison
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

try:
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ml_models.data_preparation import DataPreparation
from ml_models.lstm_model import LSTMPredictor
from ml_models.ensemble_model import EnsemblePredictor
from utils.logger import logger


class ModelTrainer:
    """
    Coordinates training of ML models
    """
    
    def __init__(self, data_prep: DataPreparation):
        """
        Initialize model trainer
        
        Args:
            data_prep: DataPreparation instance
        """
        self.data_prep = data_prep
        self.lstm_model = None
        self.ensemble_model = None
        self.results = {}
        
        logger.info("ModelTrainer initialized")
    
    def train_lstm(self, df: pd.DataFrame, symbol: str,
                   sequence_length: int = 60,
                   lstm_units: List[int] = [50, 50],
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001,
                   epochs: int = 100,
                   batch_size: int = 32,
                   validation_split: float = 0.15,
                   test_split: float = 0.15,
                   target_col: str = 'close',
                   target_type: str = 'price',
                   bidirectional: bool = False) -> Dict:
        """
        Train LSTM model
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            sequence_length: Length of input sequences
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            test_split: Test split ratio
            target_col: Target column name
            target_type: Target type ('price', 'return', 'direction')
            bidirectional: Use bidirectional LSTM
            
        Returns:
            Training results
        """
        logger.info(f"Training LSTM model for {symbol}")
        
        # Prepare features
        df_features = self.data_prep.prepare_features(df)
        
        # Create target
        df_features = self.data_prep.create_target(
            df_features,
            target_col=target_col,
            target_type=target_type
        )
        
        # Scale features
        df_scaled, scaler = self.data_prep.scale_features(df_features)
        
        # Create sequences
        X, y = self.data_prep.create_sequences(
            df_scaled,
            sequence_length=sequence_length
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_prep.split_data(
            X, y,
            validation_split=validation_split,
            test_split=test_split
        )
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Initialize LSTM model
        self.lstm_model = LSTMPredictor(
            sequence_length=sequence_length,
            n_features=X_train.shape[2]
        )
        
        # Build model
        self.lstm_model.build_model(
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            bidirectional=bidirectional
        )
        
        # Train model
        history = self.lstm_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate on test set
        test_metrics = self.lstm_model.evaluate(X_test, y_test)
        
        # Save results
        results = {
            'symbol': symbol,
            'model_type': 'LSTM',
            'sequence_length': sequence_length,
            'n_features': X_train.shape[2],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'history': history,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['lstm'] = results
        
        # Save model
        model_path = f'data/models/lstm_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        self.lstm_model.save_model(model_path)
        results['model_path'] = model_path
        
        logger.info(f"LSTM training completed. Test RMSE: {test_metrics['rmse']:.4f}")
        
        return results
    
    def train_ensemble(self, df: pd.DataFrame, symbol: str,
                      test_split: float = 0.15,
                      validation_split: float = 0.15,
                      target_col: str = 'close',
                      target_type: str = 'price',
                      use_xgboost: bool = True) -> Dict:
        """
        Train ensemble model
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            test_split: Test split ratio
            validation_split: Validation split ratio
            target_col: Target column name
            target_type: Target type
            use_xgboost: Include XGBoost in ensemble
            
        Returns:
            Training results
        """
        logger.info(f"Training ensemble model for {symbol}")
        
        # Prepare features
        df_features = self.data_prep.prepare_features(df)
        
        # Create target
        df_features = self.data_prep.create_target(
            df_features,
            target_col=target_col,
            target_type=target_type
        )
        
        # Remove NaN values
        df_features = df_features.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df_features.columns if col not in ['target', 'timestamp', 'symbol']]
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # Split data (no sequences for traditional ML)
        total_samples = len(X)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - test_size - val_size
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Initialize ensemble model
        self.ensemble_model = EnsemblePredictor()
        
        # Build models
        self.ensemble_model.build_models(use_xgboost=use_xgboost)
        
        # Train individual models
        train_results = self.ensemble_model.train(X_train, y_train, X_val, y_val)
        
        # Build and train voting ensemble
        self.ensemble_model.train_ensemble(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = self.ensemble_model.evaluate(X_test, y_test, use_ensemble=True)
        
        # Get feature importance
        feature_importance = self.ensemble_model.get_feature_importance(feature_names=feature_cols)
        
        # Compare models
        comparison = self.ensemble_model.compare_models(X_test, y_test)
        
        # Save results
        results = {
            'symbol': symbol,
            'model_type': 'Ensemble',
            'n_features': X_train.shape[1],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_results': train_results,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model_comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['ensemble'] = results
        
        # Save models
        model_dir = f'data/models/ensemble_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.ensemble_model.save_models(model_dir)
        results['model_dir'] = model_dir
        
        logger.info(f"Ensemble training completed. Test RMSE: {test_metrics['rmse']:.4f}")
        
        return results
    
    def cross_validate_ensemble(self, df: pd.DataFrame, n_splits: int = 5,
                                target_col: str = 'close',
                                target_type: str = 'price') -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            df: Stock data DataFrame
            n_splits: Number of CV splits
            target_col: Target column
            target_type: Target type
            
        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for cross-validation")
            return {}
        
        logger.info(f"Performing {n_splits}-fold time series cross-validation")
        
        # Prepare features
        df_features = self.data_prep.prepare_features(df)
        df_features = self.data_prep.create_target(df_features, target_col, target_type)
        df_features = df_features.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df_features.columns if col not in ['target', 'timestamp', 'symbol']]
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {}
        
        # Build models for CV
        temp_ensemble = EnsemblePredictor()
        temp_ensemble.build_models()
        
        for model_name, model in temp_ensemble.models.items():
            logger.info(f"Cross-validating {model_name}")
            
            scores = {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': []}
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
                test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
                
                train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
                test_r2 = 1 - np.sum((y_test - test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
                
                scores['train_rmse'].append(train_rmse)
                scores['test_rmse'].append(test_rmse)
                scores['train_r2'].append(train_r2)
                scores['test_r2'].append(test_r2)
                
                logger.info(f"  Fold {fold}: Test RMSE={test_rmse:.4f}, Test R2={test_r2:.4f}")
            
            # Calculate mean and std
            cv_results[model_name] = {
                'train_rmse_mean': np.mean(scores['train_rmse']),
                'train_rmse_std': np.std(scores['train_rmse']),
                'test_rmse_mean': np.mean(scores['test_rmse']),
                'test_rmse_std': np.std(scores['test_rmse']),
                'train_r2_mean': np.mean(scores['train_r2']),
                'train_r2_std': np.std(scores['train_r2']),
                'test_r2_mean': np.mean(scores['test_r2']),
                'test_r2_std': np.std(scores['test_r2'])
            }
            
            logger.info(f"  {model_name} CV: Test RMSE = {cv_results[model_name]['test_rmse_mean']:.4f} +/- {cv_results[model_name]['test_rmse_std']:.4f}")
        
        return cv_results
    
    def predict_future(self, df: pd.DataFrame, steps: int = 5,
                      model_type: str = 'lstm') -> np.ndarray:
        """
        Predict future values
        
        Args:
            df: Stock data DataFrame
            steps: Number of steps to predict
            model_type: Type of model to use ('lstm' or 'ensemble')
            
        Returns:
            Array of predictions
        """
        logger.info(f"Predicting next {steps} steps using {model_type}")
        
        if model_type == 'lstm':
            if self.lstm_model is None:
                raise ValueError("LSTM model not trained")
            
            # Prepare last sequence
            df_features = self.data_prep.prepare_features(df)
            df_scaled, _ = self.data_prep.scale_features(df_features)
            
            # Get last sequence
            last_sequence = df_scaled.values[-self.lstm_model.sequence_length:]
            
            # Predict
            predictions = self.lstm_model.predict_next(last_sequence, steps=steps)
            
        elif model_type == 'ensemble':
            if self.ensemble_model is None:
                raise ValueError("Ensemble model not trained")
            
            # For ensemble, we'll predict one step at a time
            df_features = self.data_prep.prepare_features(df)
            predictions = []
            
            for i in range(steps):
                # Get last row features
                feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'symbol']]
                X = df_features[feature_cols].iloc[-1:].values
                
                # Predict
                pred = self.ensemble_model.predict(X, use_ensemble=True)[0]
                predictions.append(pred)
                
                # Update features with prediction (simplified)
                # In practice, you'd need to properly update all features
                df_features.loc[len(df_features)] = df_features.iloc[-1]
                df_features.iloc[-1]['close'] = pred
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return np.array(predictions)
    
    def save_results(self, filepath: str = 'data/models/training_results.pkl'):
        """
        Save training results
        
        Args:
            filepath: Path to save results
        """
        import pickle
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str = 'data/models/training_results.pkl'):
        """
        Load training results
        
        Args:
            filepath: Path to load results from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        
        logger.info(f"Results loaded from {filepath}")
    
    def generate_report(self) -> str:
        """
        Generate training report
        
        Returns:
            Markdown formatted report
        """
        report = "# Model Training Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # LSTM results
        if 'lstm' in self.results:
            lstm = self.results['lstm']
            report += "## LSTM Model\n\n"
            report += f"- Symbol: {lstm['symbol']}\n"
            report += f"- Sequence Length: {lstm['sequence_length']}\n"
            report += f"- Features: {lstm['n_features']}\n"
            report += f"- Training Samples: {lstm['train_samples']}\n"
            report += f"- Test Samples: {lstm['test_samples']}\n\n"
            report += "### Test Metrics\n\n"
            for metric, value in lstm['test_metrics'].items():
                report += f"- {metric}: {value:.4f}\n"
            report += f"\n- Model saved to: `{lstm.get('model_path', 'N/A')}`\n\n"
        
        # Ensemble results
        if 'ensemble' in self.results:
            ens = self.results['ensemble']
            report += "## Ensemble Model\n\n"
            report += f"- Symbol: {ens['symbol']}\n"
            report += f"- Features: {ens['n_features']}\n"
            report += f"- Training Samples: {ens['train_samples']}\n"
            report += f"- Test Samples: {ens['test_samples']}\n\n"
            report += "### Test Metrics\n\n"
            for metric, value in ens['test_metrics'].items():
                report += f"- {metric}: {value:.4f}\n"
            report += f"\n- Models saved to: `{ens.get('model_dir', 'N/A')}`\n\n"
            
            if 'model_comparison' in ens:
                report += "### Model Comparison\n\n"
                report += ens['model_comparison'].to_markdown(index=False)
                report += "\n\n"
        
        return report
