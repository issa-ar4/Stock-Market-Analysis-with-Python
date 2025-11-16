# Stock Market Analysis Dashboard

Interactive web dashboard for real-time stock analysis, technical indicators, pattern recognition, and ML predictions.

## Features

- 📈 **Interactive Price Charts**: Candlestick charts with volume analysis
- 📊 **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA
- 🔍 **Pattern Recognition**: Automatic detection of candlestick and chart patterns
- 🤖 **ML Predictions**: Ensemble and LSTM models for price prediction
- ⚡ **Real-time Updates**: Live data updates and analysis
- 🎨 **Modern UI**: Clean, responsive interface with dark theme

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly
```

### 2. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Using the Dashboard

1. **Enter Stock Symbol**: Type any stock symbol (e.g., AAPL, MSFT, GOOGL) in the sidebar
2. **Select Time Period**: Choose from 1M, 3M, 6M, 1Y, 2Y, or 5Y
3. **Enable Analysis**: Check the boxes for technical indicators, patterns, or ML predictions
4. **Explore Tabs**:
   - **Price Chart**: View candlestick charts and trading volume
   - **Technical Analysis**: See indicators like RSI, MACD, Bollinger Bands
   - **Patterns**: Discover candlestick patterns and support/resistance levels
   - **ML Predictions**: Train models and get future price predictions

## Dashboard Tabs

### 📈 Price Chart
- Interactive candlestick chart
- Volume analysis with color-coded bars
- Zoom and pan functionality

### 📊 Technical Analysis
- Real-time indicator calculations
- Visual signals (Oversold/Overbought, Bullish/Bearish)
- Multiple chart views

### 🔍 Patterns
- Automatic pattern detection (Doji, Hammer, Engulfing, etc.)
- Support and resistance levels
- Trend analysis (Uptrend/Downtrend/Sideways)

### 🤖 ML Predictions
- Train ensemble models (Random Forest, Gradient Boosting, etc.)
- 5-day price predictions
- Model performance metrics

## Configuration

### Customizing the Dashboard

Edit `dashboard/app.py` to customize:
- Chart colors and themes
- Indicator parameters
- Prediction timeframes
- UI layout

### Adding New Features

The dashboard is modular and easy to extend:
- Add new tabs in the `main_dashboard()` function
- Integrate new indicators from `data_analysis/technical_indicators.py`
- Add custom visualizations using Plotly

## Tips

- **Performance**: Use shorter time periods (1M, 3M) for faster loading
- **ML Training**: Training models takes 5-10 minutes; do this once per session
- **Data Refresh**: Click "🔄 Refresh Data" to get the latest data
- **Multiple Stocks**: Change the symbol in the sidebar to analyze different stocks

## Troubleshooting

### Dashboard won't start
```bash
# Make sure Streamlit is installed
pip install streamlit

# Check if the port is already in use
streamlit run dashboard/app.py --server.port 8502
```

### No data available
```bash
# Fetch data first
python scripts/fetch_historical_data.py --symbols AAPL,MSFT,GOOGL
```

### ML predictions not working
- Ensure TensorFlow is installed: `pip install tensorflow`
- Training models requires several minutes
- Check logs for detailed error messages

## Advanced Usage

### Running on a Server

```bash
# Run on a specific IP and port
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

### Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

## Screenshots

The dashboard includes:
- Clean, modern interface with dark theme
- Real-time metrics at the top (price, change %, volume)
- Tabbed navigation for different analysis views
- Interactive Plotly charts with zoom/pan
- One-click ML model training

## Next Steps

- Explore different stocks and time periods
- Experiment with technical indicators
- Train ML models for predictions
- Compare multiple stocks using the comparison chart
- Export data and reports
