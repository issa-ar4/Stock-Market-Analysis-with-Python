"""
Visualization Module
Creates interactive charts and visualizations using Plotly
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from utils.logger import logger


class StockVisualizer:
    """
    Stock data visualization class using Plotly
    """
    
    def __init__(self, df: pd.DataFrame, symbol: str = "Stock"):
        """
        Initialize visualizer with price data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for titles
        """
        self.df = df.copy()
        self.symbol = symbol
        logger.info(f"StockVisualizer initialized for {symbol}")
    
    def candlestick_chart(self, title: str = None, show_volume: bool = True,
                         indicators: Dict = None) -> go.Figure:
        """
        Create candlestick chart with optional volume and indicators
        
        Args:
            title: Chart title
            show_volume: Whether to show volume subplot
            indicators: Dict of indicator data to overlay
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating candlestick chart for {self.symbol}")
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title or f'{self.symbol} Price Chart', 'Volume')
            )
        else:
            fig = go.Figure()
        
        # Add candlestick
        candlestick = go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add indicators if provided
        if indicators:
            for name, data in indicators.items():
                if isinstance(data, pd.Series):
                    trace = go.Scatter(
                        x=data.index,
                        y=data.values,
                        name=name,
                        mode='lines'
                    )
                    if show_volume:
                        fig.add_trace(trace, row=1, col=1)
                    else:
                        fig.add_trace(trace)
        
        # Add volume bars
        if show_volume:
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(self.df['close'], self.df['open'])]
            
            volume_bar = go.Bar(
                x=self.df.index,
                y=self.df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            )
            fig.add_trace(volume_bar, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title or f'{self.symbol} Candlestick Chart',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            hovermode='x unified',
            height=700
        )
        
        if show_volume:
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def line_chart(self, columns: List[str] = None, title: str = None) -> go.Figure:
        """
        Create line chart for specified columns
        
        Args:
            columns: List of column names to plot
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating line chart for {self.symbol}")
        
        if columns is None:
            columns = ['close']
        
        fig = go.Figure()
        
        for col in columns:
            if col in self.df.columns:
                fig.add_trace(go.Scatter(
                    x=self.df.index,
                    y=self.df[col],
                    name=col.upper(),
                    mode='lines'
                ))
        
        fig.update_layout(
            title=title or f'{self.symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def bollinger_bands_chart(self, upper: pd.Series, middle: pd.Series, 
                             lower: pd.Series, title: str = None) -> go.Figure:
        """
        Create Bollinger Bands chart
        
        Args:
            upper: Upper band data
            middle: Middle band data
            lower: Lower band data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating Bollinger Bands chart for {self.symbol}")
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['close'],
            name='Close Price',
            line=dict(color='white', width=2)
        ))
        
        # Add upper band
        fig.add_trace(go.Scatter(
            x=upper.index,
            y=upper,
            name='Upper Band',
            line=dict(color='rgba(250, 128, 114, 0.5)', width=1)
        ))
        
        # Add middle band
        fig.add_trace(go.Scatter(
            x=middle.index,
            y=middle,
            name='Middle Band (SMA)',
            line=dict(color='rgba(173, 216, 230, 0.8)', width=1, dash='dash')
        ))
        
        # Add lower band with fill
        fig.add_trace(go.Scatter(
            x=lower.index,
            y=lower,
            name='Lower Band',
            line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(250, 128, 114, 0.1)'
        ))
        
        fig.update_layout(
            title=title or f'{self.symbol} Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def rsi_chart(self, rsi: pd.Series, title: str = None) -> go.Figure:
        """
        Create RSI indicator chart
        
        Args:
            rsi: RSI data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating RSI chart for {self.symbol}")
        
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=rsi.index,
            y=rsi,
            name='RSI',
            line=dict(color='#2196F3', width=2)
        ))
        
        # Add overbought line (70)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            annotation_text="Overbought (70)",
            annotation_position="right"
        )
        
        # Add oversold line (30)
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Oversold (30)",
            annotation_position="right"
        )
        
        # Add neutral line (50)
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            annotation_text="Neutral (50)",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=title or f'{self.symbol} RSI Indicator',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            template='plotly_dark',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def macd_chart(self, macd_line: pd.Series, signal_line: pd.Series,
                  histogram: pd.Series, title: str = None) -> go.Figure:
        """
        Create MACD indicator chart
        
        Args:
            macd_line: MACD line data
            signal_line: Signal line data
            histogram: MACD histogram data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating MACD chart for {self.symbol}")
        
        fig = go.Figure()
        
        # Add MACD histogram
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in histogram]
        fig.add_trace(go.Bar(
            x=histogram.index,
            y=histogram,
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.5
        ))
        
        # Add MACD line
        fig.add_trace(go.Scatter(
            x=macd_line.index,
            y=macd_line,
            name='MACD Line',
            line=dict(color='#2196F3', width=2)
        ))
        
        # Add Signal line
        fig.add_trace(go.Scatter(
            x=signal_line.index,
            y=signal_line,
            name='Signal Line',
            line=dict(color='#FF9800', width=2)
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray"
        )
        
        fig.update_layout(
            title=title or f'{self.symbol} MACD Indicator',
            xaxis_title='Date',
            yaxis_title='MACD',
            template='plotly_dark',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def volume_chart(self, title: str = None) -> go.Figure:
        """
        Create volume chart
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating volume chart for {self.symbol}")
        
        # Color bars based on price movement
        colors = ['#26a69a' if close >= open else '#ef5350' 
                 for close, open in zip(self.df['close'], self.df['open'])]
        
        fig = go.Figure(data=[go.Bar(
            x=self.df.index,
            y=self.df['volume'],
            name='Volume',
            marker_color=colors
        )])
        
        fig.update_layout(
            title=title or f'{self.symbol} Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_dark',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def multi_indicator_chart(self, indicators: Dict[str, pd.Series]) -> go.Figure:
        """
        Create comprehensive chart with price, volume, and multiple indicators
        
        Args:
            indicators: Dictionary of indicator names and data
            
        Returns:
            Plotly Figure object with multiple subplots
        """
        logger.info(f"Creating multi-indicator chart for {self.symbol}")
        
        # Create subplots
        num_indicators = len(indicators)
        rows = 2 + num_indicators
        row_heights = [0.4, 0.15] + [0.45/num_indicators] * num_indicators
        
        subplot_titles = [f'{self.symbol} Price', 'Volume'] + list(indicators.keys())
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['open'],
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add volume
        colors = ['#26a69a' if close >= open else '#ef5350' 
                 for close, open in zip(self.df['close'], self.df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add indicators
        for i, (name, data) in enumerate(indicators.items(), start=3):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data,
                    name=name,
                    mode='lines',
                    showlegend=False
                ),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Technical Analysis Dashboard',
            template='plotly_dark',
            hovermode='x unified',
            height=300 * rows,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def comparison_chart(self, other_dfs: Dict[str, pd.DataFrame], 
                        column: str = 'close', normalize: bool = True,
                        title: str = None) -> go.Figure:
        """
        Create comparison chart for multiple stocks
        
        Args:
            other_dfs: Dictionary of {symbol: DataFrame} to compare
            column: Column to compare
            normalize: Whether to normalize to starting price
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info(f"Creating comparison chart")
        
        fig = go.Figure()
        
        # Add current stock
        data = self.df[column].copy()
        if normalize:
            data = (data / data.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data,
            name=self.symbol,
            mode='lines'
        ))
        
        # Add other stocks
        for symbol, df in other_dfs.items():
            data = df[column].copy()
            if normalize:
                data = (data / data.iloc[0]) * 100
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data,
                name=symbol,
                mode='lines'
            ))
        
        y_label = 'Normalized Price (%)' if normalize else 'Price'
        
        fig.update_layout(
            title=title or 'Stock Price Comparison',
            xaxis_title='Date',
            yaxis_title=y_label,
            template='plotly_dark',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def heatmap_correlation(self, other_dfs: Dict[str, pd.DataFrame],
                           column: str = 'close', title: str = None) -> go.Figure:
        """
        Create correlation heatmap for multiple stocks
        
        Args:
            other_dfs: Dictionary of {symbol: DataFrame} to correlate
            column: Column to use for correlation
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        logger.info("Creating correlation heatmap")
        
        # Combine all data
        combined_data = pd.DataFrame({self.symbol: self.df[column]})
        
        for symbol, df in other_dfs.items():
            combined_data[symbol] = df[column]
        
        # Calculate correlation matrix
        corr_matrix = combined_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title or 'Stock Price Correlation Matrix',
            template='plotly_dark',
            height=600,
            width=700
        )
        
        return fig


def create_summary_table(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create summary statistics table
    
    Args:
        df: Price DataFrame
        symbol: Stock symbol
        
    Returns:
        Plotly Figure object with table
    """
    logger.info(f"Creating summary table for {symbol}")
    
    # Calculate statistics
    current_price = df['close'].iloc[-1]
    open_price = df['open'].iloc[-1]
    high_price = df['high'].max()
    low_price = df['low'].min()
    avg_volume = df['volume'].mean()
    price_change = current_price - df['close'].iloc[0]
    pct_change = (price_change / df['close'].iloc[0]) * 100
    
    # Create table data
    metrics = [
        'Current Price',
        'Open Price',
        'Period High',
        'Period Low',
        'Avg Volume',
        'Price Change',
        'Percent Change'
    ]
    
    values = [
        f'${current_price:.2f}',
        f'${open_price:.2f}',
        f'${high_price:.2f}',
        f'${low_price:.2f}',
        f'{avg_volume:,.0f}',
        f'${price_change:+.2f}',
        f'{pct_change:+.2f}%'
    ]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#1e1e1e',
            align='left',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[metrics, values],
            fill_color='#2d2d2d',
            align='left',
            font=dict(color='white', size=12),
            height=30
        )
    )])
    
    fig.update_layout(
        title=f'{symbol} Summary Statistics',
        template='plotly_dark',
        height=350
    )
    
    return fig
