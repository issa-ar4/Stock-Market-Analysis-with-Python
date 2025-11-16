"""
Interactive Web Dashboard using Streamlit
Real-time stock analysis and ML predictions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from config.config import Config
from database import get_engine, get_session
from database.repositories import StockPriceRepository, StockInfoRepository
from data_ingestion.data_fetcher import StockDataFetcher
from data_analysis.technical_indicators import TechnicalAnalysis
from data_analysis.pattern_recognition import PatternRecognition
from data_analysis.visualization import StockVisualizer
from ml_models.data_preparation import DataPreparation
from ml_models.model_trainer import ModelTrainer


# Page config
st.set_page_config(
    page_title="Stock Market Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def init_components():
    """Initialize all components"""
    config = Config()
    
    # Initialize database
    db_url = Config.DATABASE_URL
    engine = get_engine(db_url)
    session = get_session(engine)
    
    stock_repo = StockPriceRepository(session)
    info_repo = StockInfoRepository(session)
    data_fetcher = StockDataFetcher()
    
    return config, stock_repo, info_repo, data_fetcher


def load_stock_data(symbol: str, stock_repo, days: int = 365):
    """Load stock data from database"""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stock_data = stock_repo.get_by_symbol(symbol=symbol, start_date=start_date, end_date=end_date)
    
    if not stock_data:
        return None
    
    df = pd.DataFrame([{
        'timestamp': record.timestamp,
        'open': float(record.open),
        'high': float(record.high),
        'low': float(record.low),
        'close': float(record.close),
        'volume': int(record.volume)
    } for record in stock_data])
    
    return df.sort_values('timestamp').reset_index(drop=True)


def sidebar():
    """Render sidebar"""
    st.sidebar.title("📊 Stock Analysis")
    
    # Symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    # Time period
    period = st.sidebar.selectbox(
        "Time Period",
        ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        index=3
    )
    
    period_map = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
        "5Y": 1825
    }
    
    days = period_map[period]
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_indicators = st.sidebar.checkbox("Technical Indicators", value=True)
    show_patterns = st.sidebar.checkbox("Pattern Recognition", value=True)
    show_predictions = st.sidebar.checkbox("ML Predictions", value=False)
    
    # Refresh data
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    return symbol, days, show_indicators, show_patterns, show_predictions


def main_dashboard(symbol, df, show_indicators, show_patterns, show_predictions):
    """Render main dashboard"""
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title(f"{symbol} Stock Analysis")
    
    with col2:
        if len(df) > 0:
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            change = last_price - prev_price
            change_pct = (change / prev_price) * 100
            
            st.metric(
                "Current Price",
                f"${last_price:.2f}",
                f"{change:+.2f} ({change_pct:+.2f}%)"
            )
    
    with col3:
        if len(df) > 0:
            avg_volume = df['volume'].tail(20).mean()
            st.metric("Avg Volume (20d)", f"{avg_volume:,.0f}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🔍 Patterns", "🤖 ML Predictions"])
    
    # Tab 1: Price Chart
    with tab1:
        visualizer = StockVisualizer(df, symbol=symbol)
        
        # Main chart
        fig = visualizer.candlestick_chart(title=f"{symbol} Price Chart")
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_volume = go.Figure()
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig_volume.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_volume.update_layout(
            title='Trading Volume',
            template='plotly_dark',
            height=300,
            xaxis_title='Date',
            yaxis_title='Volume'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Tab 2: Technical Analysis
    with tab2:
        if show_indicators:
            st.subheader("Technical Indicators")
            
            ta = TechnicalAnalysis(df)
            
            # Calculate indicators
            df_with_indicators = df.copy()
            df_with_indicators['sma_20'] = ta.sma(period=20)
            df_with_indicators['ema_20'] = ta.ema(period=20)
            df_with_indicators['rsi'] = ta.rsi(period=14)
            
            macd_line, signal_line, histogram = ta.macd()
            df_with_indicators['macd'] = macd_line
            df_with_indicators['macd_signal'] = signal_line
            df_with_indicators['macd_histogram'] = histogram
            
            bb_upper, bb_middle, bb_lower = ta.bollinger_bands()
            df_with_indicators['bb_upper'] = bb_upper
            df_with_indicators['bb_middle'] = bb_middle
            df_with_indicators['bb_lower'] = bb_lower
            
            # Indicators metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = df_with_indicators['rsi'].iloc[-1]
                rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                st.metric("RSI (14)", f"{rsi:.2f}", rsi_signal)
            
            with col2:
                macd = df_with_indicators['macd'].iloc[-1]
                signal = df_with_indicators['macd_signal'].iloc[-1]
                macd_signal = "Bullish" if macd > signal else "Bearish"
                st.metric("MACD", f"{macd:.2f}", macd_signal)
            
            with col3:
                sma_20 = df_with_indicators['sma_20'].iloc[-1]
                price = df_with_indicators['close'].iloc[-1]
                sma_signal = "Above SMA" if price > sma_20 else "Below SMA"
                st.metric("SMA (20)", f"${sma_20:.2f}", sma_signal)
            
            with col4:
                ema_20 = df_with_indicators['ema_20'].iloc[-1]
                ema_signal = "Above EMA" if price > ema_20 else "Below EMA"
                st.metric("EMA (20)", f"${ema_20:.2f}", ema_signal)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bb = visualizer.bollinger_bands_chart(
                    bb_upper, bb_middle, bb_lower, 
                    title="Bollinger Bands"
                )
                st.plotly_chart(fig_bb, use_container_width=True)
            
            with col2:
                fig_rsi = visualizer.rsi_chart(df_with_indicators['rsi'], title="RSI Indicator")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            fig_macd = visualizer.macd_chart(
                macd_line, signal_line, histogram,
                title="MACD Indicator"
            )
            st.plotly_chart(fig_macd, use_container_width=True)
    
    # Tab 3: Pattern Recognition
    with tab3:
        if show_patterns:
            st.subheader("Pattern Recognition")
            
            pr = PatternRecognition(df)
            
            # Detect patterns
            df_patterns = df.copy()
            df_patterns['doji'] = pr.doji()
            df_patterns['hammer'] = pr.hammer()
            df_patterns['engulfing_bullish'] = pr.engulfing_bullish()
            df_patterns['engulfing_bearish'] = pr.engulfing_bearish()
            
            # Show recent patterns
            pattern_cols = [col for col in df_patterns.columns if col.startswith('pattern_')]
            recent_patterns = df_patterns[['timestamp', 'close'] + pattern_cols].tail(20)
            
            # Count patterns
            pattern_counts = {}
            for col in pattern_cols:
                count = df_patterns[col].tail(50).sum()
                if count > 0:
                    pattern_name = col.replace('pattern_', '').replace('_', ' ').title()
                    pattern_counts[pattern_name] = int(count)
            
            if pattern_counts:
                st.write("**Patterns Detected (Last 50 days):**")
                
                cols = st.columns(len(pattern_counts))
                for col, (pattern, count) in zip(cols, pattern_counts.items()):
                    col.metric(pattern, count)
            else:
                st.info("No significant patterns detected in recent data")
            
            # Support and Resistance (simplified calculation)
            recent_low = df['low'].tail(50).min()
            recent_high = df['high'].tail(50).max()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Support Level (50d)", f"${recent_low:.2f}")
            with col2:
                st.metric("Resistance Level (50d)", f"${recent_high:.2f}")
            
            # Trend detection (simplified)
            sma_50 = df['close'].tail(50).mean()
            current_price = df['close'].iloc[-1]
            if current_price > sma_50 * 1.02:
                trend = "uptrend"
                trend_emoji = "📈"
            elif current_price < sma_50 * 0.98:
                trend = "downtrend"
                trend_emoji = "📉"
            else:
                trend = "sideways"
                trend_emoji = "➡️"
            st.info(f"{trend_emoji} Current Trend: **{trend.upper()}**")
    
    # Tab 4: ML Predictions
    with tab4:
        if show_predictions:
            st.subheader("Machine Learning Predictions")
            
            st.info("⚠️ Note: Training ML models takes several minutes. Models need to be trained first.")
            
            if st.button("Train ML Models"):
                with st.spinner("Training models... This may take 5-10 minutes"):
                    try:
                        # Initialize with DataFrame
                        data_prep = DataPreparation(df, target_column='close')
                        trainer = ModelTrainer(data_prep)
                        
                        # Train ensemble (faster)
                        st.write("Training Ensemble Model...")
                        ensemble_results = trainer.train_ensemble(symbol, use_xgboost=False)
                        
                        st.success("✅ Ensemble Model Trained!")
                        st.write(f"Test RMSE: {ensemble_results['test_metrics']['rmse']:.4f}")
                        st.write(f"Direction Accuracy: {ensemble_results['test_metrics']['direction_accuracy']:.2f}%")
                        
                        # Make predictions
                        predictions = trainer.predict_future(steps=5, model_type='ensemble')
                        
                        st.write("**Predictions (Next 5 Days):**")
                        pred_df = pd.DataFrame({
                            'Day': range(1, 6),
                            'Predicted Price': [f"${p:.2f}" for p in predictions]
                        })
                        st.dataframe(pred_df)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.write("**Available Models:**")
            st.write("- Ensemble (Random Forest, Gradient Boosting, Ridge, Lasso)")
            st.write("- LSTM Neural Network (Deep Learning)")
        else:
            st.info("Enable 'ML Predictions' in the sidebar to access this feature")


def main():
    """Main application"""
    
    # Initialize
    config, stock_repo, info_repo, data_fetcher = init_components()
    
    # Sidebar
    symbol, days, show_indicators, show_patterns, show_predictions = sidebar()
    
    # Load data
    with st.spinner(f"Loading data for {symbol}..."):
        df = load_stock_data(symbol, stock_repo, days)
    
    if df is None or len(df) == 0:
        st.error(f"No data available for {symbol}. Please fetch data first using `fetch_historical_data.py`")
        st.info("Run: `python scripts/fetch_historical_data.py --symbols AAPL,MSFT,GOOGL`")
        return
    
    # Render dashboard
    main_dashboard(symbol, df, show_indicators, show_patterns, show_predictions)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
