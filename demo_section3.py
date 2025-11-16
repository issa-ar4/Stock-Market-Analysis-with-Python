"""
Demo script for Section 3: ML Models
Demonstrates LSTM and Ensemble model training and prediction
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from database.repositories import StockPriceRepository
from ml_models.data_preparation import DataPreparation
from ml_models.model_trainer import ModelTrainer
from utils.logger import logger
import pandas as pd
import numpy as np


def demo_ml_models():
    """Demonstrate ML model training and prediction"""
    
    print("=" * 80)
    print("Section 3 Demo: ML Models for Stock Prediction")
    print("=" * 80)
    
    # Initialize
    config = Config()
    stock_repo = StockPriceRepository()
    
    # Fetch historical data for training
    symbol = 'AAPL'
    print(f"\n1. Fetching historical data for {symbol}...")
    
    stock_data = stock_repo.get_stock_prices(
        symbol=symbol,
        limit=1000  # Get last 1000 days
    )
    
    if not stock_data:
        print(f"No data found for {symbol}. Please run fetch_historical_data.py first.")
        return
    
    print(f"   Found {len(stock_data)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': record.timestamp,
        'open': float(record.open),
        'high': float(record.high),
        'low': float(record.low),
        'close': float(record.close),
        'volume': int(record.volume)
    } for record in stock_data])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n2. Data Overview:")
    print(f"   Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Avg Volume: {df['volume'].mean():,.0f}")
    
    # Initialize data preparation
    print(f"\n3. Preparing data for ML models...")
    data_prep = DataPreparation()
    
    # Initialize model trainer
    trainer = ModelTrainer(data_prep)
    
    # Train Ensemble Model (faster to train)
    print(f"\n4. Training Ensemble Model...")
    print(f"   This may take a few minutes...")
    
    try:
        ensemble_results = trainer.train_ensemble(
            df=df,
            symbol=symbol,
            test_split=0.15,
            validation_split=0.15,
            target_type='price',
            use_xgboost=False  # Set to True if xgboost is installed
        )
        
        print(f"\n   Ensemble Model Results:")
        print(f"   - Models: {', '.join(ensemble_results['train_results'].keys())}")
        print(f"   - Training samples: {ensemble_results['train_samples']}")
        print(f"   - Test RMSE: {ensemble_results['test_metrics']['rmse']:.4f}")
        print(f"   - Test MAE: {ensemble_results['test_metrics']['mae']:.4f}")
        print(f"   - Test R²: {ensemble_results['test_metrics']['r2']:.4f}")
        print(f"   - Direction Accuracy: {ensemble_results['test_metrics']['direction_accuracy']:.2f}%")
        
        # Show model comparison
        if 'model_comparison' in ensemble_results:
            print(f"\n   Model Comparison:")
            comparison = ensemble_results['model_comparison']
            for _, row in comparison.iterrows():
                print(f"   - {row['model']}: RMSE={row['rmse']:.4f}, R²={row['r2']:.4f}")
        
        # Show top features
        if ensemble_results['feature_importance'] is not None:
            print(f"\n   Top 10 Most Important Features:")
            top_features = ensemble_results['feature_importance'].head(10)
            for feature in top_features.index:
                importance = top_features.loc[feature, 'average']
                print(f"   - {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"   Error training ensemble: {str(e)}")
        logger.error(f"Ensemble training error: {str(e)}", exc_info=True)
    
    # Train LSTM Model (slower, requires TensorFlow)
    print(f"\n5. Training LSTM Model...")
    print(f"   This will take longer (5-10 minutes)...")
    print(f"   Using smaller sequence length for demo...")
    
    try:
        lstm_results = trainer.train_lstm(
            df=df,
            symbol=symbol,
            sequence_length=30,  # Smaller for faster training
            lstm_units=[50, 25],  # Smaller for faster training
            epochs=50,  # Fewer epochs for demo
            batch_size=32,
            target_type='price'
        )
        
        print(f"\n   LSTM Model Results:")
        print(f"   - Sequence length: {lstm_results['sequence_length']}")
        print(f"   - Features: {lstm_results['n_features']}")
        print(f"   - Training samples: {lstm_results['train_samples']}")
        print(f"   - Test RMSE: {lstm_results['test_metrics']['rmse']:.4f}")
        print(f"   - Test MAE: {lstm_results['test_metrics']['mae']:.4f}")
        print(f"   - Test R²: {lstm_results['test_metrics']['r2']:.4f}")
        print(f"   - Direction Accuracy: {lstm_results['test_metrics']['direction_accuracy']:.2f}%")
        
        # Final training metrics
        final_loss = lstm_results['history']['loss'][-1]
        final_val_loss = lstm_results['history'].get('val_loss', [None])[-1]
        print(f"   - Final train loss: {final_loss:.6f}")
        if final_val_loss:
            print(f"   - Final val loss: {final_val_loss:.6f}")
        
    except Exception as e:
        print(f"   Error training LSTM: {str(e)}")
        print(f"   Note: LSTM requires TensorFlow to be installed")
        logger.error(f"LSTM training error: {str(e)}", exc_info=True)
    
    # Make future predictions
    print(f"\n6. Making Future Predictions...")
    
    try:
        # Predict next 5 days with ensemble
        if trainer.ensemble_model:
            predictions = trainer.predict_future(df, steps=5, model_type='ensemble')
            
            print(f"\n   Ensemble Predictions (next 5 days):")
            last_price = df['close'].iloc[-1]
            print(f"   - Current price: ${last_price:.2f}")
            
            for i, pred in enumerate(predictions, 1):
                change = ((pred - last_price) / last_price) * 100
                print(f"   - Day {i}: ${pred:.2f} ({change:+.2f}%)")
                last_price = pred
        
        # Predict with LSTM if available
        if trainer.lstm_model:
            predictions_lstm = trainer.predict_future(df, steps=5, model_type='lstm')
            
            print(f"\n   LSTM Predictions (next 5 days):")
            last_price = df['close'].iloc[-1]
            print(f"   - Current price: ${last_price:.2f}")
            
            for i, pred in enumerate(predictions_lstm, 1):
                change = ((pred - last_price) / last_price) * 100
                print(f"   - Day {i}: ${pred:.2f} ({change:+.2f}%)")
                last_price = pred
                
    except Exception as e:
        print(f"   Error making predictions: {str(e)}")
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
    
    # Generate report
    print(f"\n7. Generating Training Report...")
    report = trainer.generate_report()
    
    report_path = 'data/models/training_report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"   Report saved to: {report_path}")
    
    # Save results
    trainer.save_results()
    print(f"   Results saved to: data/models/training_results.pkl")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Check the training report at: data/models/training_report.md")
    print("2. View saved models in: data/models/")
    print("3. Experiment with different hyperparameters")
    print("4. Try different stocks and timeframes")
    print("5. Move on to Section 4: Dashboard and Visualization")
    print("=" * 80)


if __name__ == "__main__":
    try:
        demo_ml_models()
    except Exception as e:
        logger.error(f"Demo error: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("Make sure you have:")
        print("1. Installed all dependencies (pip install -r requirements.txt)")
        print("2. Set up the database (python scripts/init_db.py)")
        print("3. Fetched historical data (python scripts/fetch_historical_data.py)")
