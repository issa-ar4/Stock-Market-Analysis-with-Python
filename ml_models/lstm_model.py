"""
LSTM Model for Stock Price Prediction
Uses TensorFlow/Keras for building and training LSTM networks
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import pickle
import os

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy types for type hints
    Sequential = type(None)
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

from utils.logger import logger


class LSTMPredictor:
    """
    LSTM-based stock price predictor
    """
    
    def __init__(self, sequence_length: int = 60, n_features: int = None):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler = None
        
        logger.info(f"LSTMPredictor initialized (sequence_length={sequence_length})")
    
    def build_model(self, lstm_units: list = [50, 50], dropout_rate: float = 0.2,
                   learning_rate: float = 0.001, bidirectional: bool = False):
        """
        Build LSTM model architecture
        
        Args:
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional LSTM
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model")
        
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")
        
        model = Sequential()
        
        # First LSTM layer
        if bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1),
                input_shape=(self.sequence_length, self.n_features)
            ))
        else:
            model.add(LSTM(
                lstm_units[0],
                return_sequences=len(lstm_units) > 1,
                input_shape=(self.sequence_length, self.n_features)
            ))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        
        # Print model summary
        logger.info(f"Model built with {model.count_params():,} parameters")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 100, batch_size: int = 32,
             early_stopping_patience: int = 10,
             reduce_lr_patience: int = 5) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training LSTM model (epochs={epochs}, batch_size={batch_size})")
        
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = 'data/models/lstm_checkpoint.h5'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        logger.info("Training completed")
        logger.info(f"Final train loss: {history.history['loss'][-1]:.6f}")
        if validation_data:
            logger.info(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X, verbose=0)
        
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Model metrics
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate additional metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Direction accuracy (for price prediction)
        direction_actual = np.sign(np.diff(y_test, prepend=y_test[0]))
        direction_pred = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'history': self.history
        }
        
        config_path = filepath.replace('.h5', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        # Load Keras model
        self.model = load_model(filepath)
        
        # Load configuration
        config_path = filepath.replace('.h5', '_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            self.sequence_length = config['sequence_length']
            self.n_features = config['n_features']
            self.history = config.get('history')
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict_next(self, last_sequence: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict next n steps
        
        Args:
            last_sequence: Last sequence of data
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        logger.info(f"Predicting next {steps} steps")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, self.n_features), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence (simple approach: shift and append prediction)
            # Note: This is simplified - in practice, you'd need to update all features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred[0, 0]  # Update first feature (price)
        
        return np.array(predictions)
    
    def plot_training_history(self):
        """
        Plot training history
        
        Returns:
            Plotly figure
        """
        if self.history is None:
            logger.warning("No training history available")
            return None
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Loss', 'MAE'),
                vertical_spacing=0.1
            )
            
            epochs = list(range(1, len(self.history['loss']) + 1))
            
            # Loss plot
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history['loss'], name='Train Loss', mode='lines'),
                row=1, col=1
            )
            if 'val_loss' in self.history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=self.history['val_loss'], name='Val Loss', mode='lines'),
                    row=1, col=1
                )
            
            # MAE plot
            fig.add_trace(
                go.Scatter(x=epochs, y=self.history['mae'], name='Train MAE', mode='lines'),
                row=2, col=1
            )
            if 'val_mae' in self.history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=self.history['val_mae'], name='Val MAE', mode='lines'),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="MAE", row=2, col=1)
            
            fig.update_layout(
                title='Training History',
                template='plotly_dark',
                height=600
            )
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available for plotting")
            return None
