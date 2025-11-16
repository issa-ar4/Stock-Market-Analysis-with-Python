"""
Ensemble Model combining multiple ML approaches
Uses Random Forest, XGBoost, and other models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
import os

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from utils.logger import logger


class EnsemblePredictor:
    """
    Ensemble model combining multiple ML algorithms
    """
    
    def __init__(self):
        """Initialize ensemble predictor"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ensemble models")
        
        self.models = {}
        self.ensemble = None
        self.feature_importances = None
        
        logger.info("EnsemblePredictor initialized")
    
    def build_models(self, use_xgboost: bool = True) -> Dict:
        """
        Build individual models for ensemble
        
        Args:
            use_xgboost: Whether to include XGBoost
            
        Returns:
            Dictionary of models
        """
        logger.info("Building ensemble models")
        
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # XGBoost
        if use_xgboost and XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # Ridge Regression
        models['ridge'] = Ridge(alpha=1.0)
        
        # Lasso Regression
        models['lasso'] = Lasso(alpha=0.1, max_iter=10000)
        
        self.models = models
        
        logger.info(f"Built {len(models)} models: {list(models.keys())}")
        
        return models
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on training set
                train_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                train_mae = mean_absolute_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                
                result = {
                    'train_rmse': train_rmse,
                    'train_mae': train_mae,
                    'train_r2': train_r2
                }
                
                # Evaluate on validation set
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                    val_mae = mean_absolute_error(y_val, val_pred)
                    val_r2 = r2_score(y_val, val_pred)
                    
                    result.update({
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'val_r2': val_r2
                    })
                    
                    logger.info(f"  {name} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
                else:
                    logger.info(f"  {name} - Train RMSE: {train_rmse:.4f}")
                
                results[name] = result
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def build_voting_ensemble(self, weights: Optional[List[float]] = None):
        """
        Build voting ensemble from trained models
        
        Args:
            weights: Optional weights for each model
        """
        logger.info("Building voting ensemble")
        
        # Create list of (name, model) tuples
        estimators = [(name, model) for name, model in self.models.items()]
        
        # Create voting regressor
        self.ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        
        logger.info(f"Voting ensemble created with {len(estimators)} models")
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the voting ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Training voting ensemble")
        
        if self.ensemble is None:
            self.build_voting_ensemble()
        
        self.ensemble.fit(X_train, y_train)
        
        logger.info("Voting ensemble trained")
    
    def predict(self, X: np.ndarray, use_ensemble: bool = True) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            use_ensemble: Use ensemble or individual models
            
        Returns:
            Predictions
        """
        if use_ensemble:
            if self.ensemble is None:
                raise ValueError("Ensemble not trained. Call train_ensemble() first.")
            
            logger.info(f"Making ensemble predictions for {len(X)} samples")
            return self.ensemble.predict(X)
        else:
            # Average predictions from all models
            logger.info(f"Making averaged predictions for {len(X)} samples")
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
    
    def predict_individual(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of predictions per model
        """
        logger.info(f"Making individual predictions for {len(X)} samples")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                use_ensemble: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            use_ensemble: Evaluate ensemble or average of models
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Make predictions
        y_pred = self.predict(X_test, use_ensemble=use_ensemble)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Direction accuracy
        direction_actual = np.sign(np.diff(y_test, prepend=y_test[0]))
        direction_pred = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame of feature importances
        """
        logger.info("Calculating feature importances")
        
        importances = {}
        
        # Get importance from each model that supports it
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        if not importances:
            logger.warning("No models with feature importance available")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(importances)
        
        if feature_names:
            df.index = feature_names
        
        # Calculate average importance
        df['average'] = df.mean(axis=1)
        df = df.sort_values('average', ascending=False)
        
        self.feature_importances = df
        
        logger.info(f"Top 10 features:\n{df.head(10)}")
        
        return df
    
    def save_models(self, directory: str):
        """
        Save all models to directory
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            filepath = os.path.join(directory, f"{name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save ensemble
        if self.ensemble is not None:
            ensemble_path = os.path.join(directory, "ensemble.pkl")
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble, f)
            logger.info(f"Saved ensemble to {ensemble_path}")
        
        # Save feature importances
        if self.feature_importances is not None:
            importance_path = os.path.join(directory, "feature_importances.csv")
            self.feature_importances.to_csv(importance_path)
            logger.info(f"Saved feature importances to {importance_path}")
    
    def load_models(self, directory: str):
        """
        Load models from directory
        
        Args:
            directory: Directory to load models from
        """
        logger.info(f"Loading models from {directory}")
        
        # Load individual models
        for filename in os.listdir(directory):
            if filename.endswith('.pkl') and filename != 'ensemble.pkl':
                name = filename.replace('.pkl', '')
                filepath = os.path.join(directory, filename)
                with open(filepath, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info(f"Loaded {name}")
        
        # Load ensemble
        ensemble_path = os.path.join(directory, "ensemble.pkl")
        if os.path.exists(ensemble_path):
            with open(ensemble_path, 'rb') as f:
                self.ensemble = pickle.load(f)
            logger.info("Loaded ensemble")
        
        # Load feature importances
        importance_path = os.path.join(directory, "feature_importances.csv")
        if os.path.exists(importance_path):
            self.feature_importances = pd.read_csv(importance_path, index_col=0)
            logger.info("Loaded feature importances")
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of all models
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame comparing model performance
        """
        logger.info("Comparing model performance")
        
        results = []
        
        # Evaluate each model
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                'model': name,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            results.append(metrics)
        
        # Evaluate ensemble if available
        if self.ensemble is not None:
            y_pred = self.ensemble.predict(X_test)
            
            metrics = {
                'model': 'ensemble',
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.sort_values('rmse')
        
        logger.info(f"Model comparison:\n{df}")
        
        return df
