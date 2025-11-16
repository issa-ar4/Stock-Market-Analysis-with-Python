"""
Unit tests for ML models
Tests data preparation, LSTM, and ensemble models
"""
import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.data_preparation import DataPreparation
from ml_models.lstm_model import LSTMPredictor
from ml_models.ensemble_model import EnsemblePredictor


class TestDataPreparation(unittest.TestCase):
    """Test data preparation module"""
    
    def setUp(self):
        """Create sample data"""
        self.data_prep = DataPreparation()
        
        # Create sample DataFrame
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    def test_add_technical_features(self):
        """Test adding technical indicators"""
        df = self.data_prep.add_technical_features(self.df.copy())
        
        # Check new columns exist
        self.assertIn('sma_20', df.columns)
        self.assertIn('rsi', df.columns)
        self.assertIn('macd', df.columns)
        
        print("✓ Technical features added successfully")
    
    def test_add_price_features(self):
        """Test adding price-based features"""
        df = self.data_prep.add_price_features(self.df.copy())
        
        # Check new columns
        self.assertIn('price_change', df.columns)
        self.assertIn('price_change_pct', df.columns)
        self.assertIn('high_low_range', df.columns)
        
        print("✓ Price features added successfully")
    
    def test_add_time_features(self):
        """Test adding time-based features"""
        df = self.data_prep.add_time_features(self.df.copy())
        
        # Check new columns
        self.assertIn('day_of_week', df.columns)
        self.assertIn('month', df.columns)
        
        print("✓ Time features added successfully")
    
    def test_add_lag_features(self):
        """Test adding lag features"""
        df = self.data_prep.add_lag_features(self.df.copy(), lags=[1, 2, 3])
        
        # Check new columns
        self.assertIn('close_lag_1', df.columns)
        self.assertIn('volume_lag_1', df.columns)
        
        print("✓ Lag features added successfully")
    
    def test_create_target(self):
        """Test target creation"""
        df = self.data_prep.create_target(self.df.copy(), target_type='price')
        
        # Check target column
        self.assertIn('target', df.columns)
        
        print("✓ Target created successfully")
    
    def test_create_sequences(self):
        """Test sequence creation for LSTM"""
        # Prepare simple data
        data = np.random.rand(100, 5)
        X, y = self.data_prep.create_sequences(pd.DataFrame(data), sequence_length=10)
        
        # Check shapes
        self.assertEqual(X.shape[0], 90)  # 100 - 10
        self.assertEqual(X.shape[1], 10)  # sequence_length
        self.assertEqual(len(y), 90)
        
        print("✓ Sequences created successfully")
    
    def test_split_data(self):
        """Test data splitting"""
        X = np.random.rand(100, 10, 5)
        y = np.random.rand(100)
        
        splits = self.data_prep.split_data(X, y, validation_split=0.15, test_split=0.15)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check splits
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_val), 15)
        self.assertEqual(len(X_test), 15)
        
        print("✓ Data split successfully")


class TestLSTMPredictor(unittest.TestCase):
    """Test LSTM model"""
    
    def setUp(self):
        """Setup LSTM predictor"""
        try:
            self.predictor = LSTMPredictor(sequence_length=10, n_features=5)
            self.has_tensorflow = True
        except ImportError:
            self.has_tensorflow = False
    
    def test_build_model(self):
        """Test model building"""
        if not self.has_tensorflow:
            self.skipTest("TensorFlow not available")
        
        model = self.predictor.build_model(lstm_units=[20, 10])
        
        # Check model exists
        self.assertIsNotNone(model)
        self.assertGreater(model.count_params(), 0)
        
        print("✓ LSTM model built successfully")
    
    def test_train_predict(self):
        """Test training and prediction"""
        if not self.has_tensorflow:
            self.skipTest("TensorFlow not available")
        
        # Create dummy data
        X_train = np.random.rand(50, 10, 5)
        y_train = np.random.rand(50)
        X_test = np.random.rand(10, 10, 5)
        
        # Build and train
        self.predictor.build_model(lstm_units=[10])
        self.predictor.train(X_train, y_train, epochs=2, batch_size=8)
        
        # Predict
        predictions = self.predictor.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), 10)
        
        print("✓ LSTM training and prediction successful")


class TestEnsemblePredictor(unittest.TestCase):
    """Test ensemble model"""
    
    def setUp(self):
        """Setup ensemble predictor"""
        try:
            self.predictor = EnsemblePredictor()
            self.has_sklearn = True
        except ImportError:
            self.has_sklearn = False
    
    def test_build_models(self):
        """Test building models"""
        if not self.has_sklearn:
            self.skipTest("scikit-learn not available")
        
        models = self.predictor.build_models(use_xgboost=False)
        
        # Check models created
        self.assertGreater(len(models), 0)
        self.assertIn('random_forest', models)
        
        print("✓ Ensemble models built successfully")
    
    def test_train_predict(self):
        """Test training and prediction"""
        if not self.has_sklearn:
            self.skipTest("scikit-learn not available")
        
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(20, 10)
        
        # Build and train
        self.predictor.build_models(use_xgboost=False)
        self.predictor.train(X_train, y_train)
        
        # Predict
        predictions = self.predictor.predict(X_test, use_ensemble=False)
        
        # Check predictions
        self.assertEqual(len(predictions), 20)
        
        print("✓ Ensemble training and prediction successful")
    
    def test_voting_ensemble(self):
        """Test voting ensemble"""
        if not self.has_sklearn:
            self.skipTest("scikit-learn not available")
        
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(20, 10)
        
        # Build, train, and create ensemble
        self.predictor.build_models(use_xgboost=False)
        self.predictor.train(X_train, y_train)
        self.predictor.build_voting_ensemble()
        self.predictor.train_ensemble(X_train, y_train)
        
        # Predict
        predictions = self.predictor.predict(X_test, use_ensemble=True)
        
        # Check predictions
        self.assertEqual(len(predictions), 20)
        
        print("✓ Voting ensemble successful")


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running ML Model Tests")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsemblePredictor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
