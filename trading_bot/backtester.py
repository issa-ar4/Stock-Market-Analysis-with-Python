"""
Backtesting Engine for Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Backtester:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005    # 0.05% slippage
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (fraction)
            slippage: Slippage rate (fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None
        self.trades = []
    
    def run(
        self,
        strategy,
        df: pd.DataFrame,
        position_size: float = 0.95  # Use 95% of capital
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            strategy: Trading strategy object
            df: DataFrame with OHLCV data
            position_size: Fraction of capital to use per trade
            
        Returns:
            Dictionary with backtest results
        """
        df = df.copy()
        
        # Generate signals
        df = strategy.generate_signals(df)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares held
        position_value = 0
        entry_price = 0
        
        # Track portfolio value over time
        portfolio_values = []
        positions_held = []
        trades = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = row['signal']
            price = row['close']
            
            # Calculate current portfolio value
            current_value = capital + (position * price)
            portfolio_values.append(current_value)
            positions_held.append(position)
            
            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                position = int((capital * position_size) / (price * (1 + self.commission + self.slippage)))
                if position > 0:
                    entry_price = price * (1 + self.slippage)
                    cost = position * entry_price * (1 + self.commission)
                    capital -= cost
                    
                    trades.append({
                        'date': row.name,
                        'type': 'BUY',
                        'price': entry_price,
                        'shares': position,
                        'cost': cost,
                        'capital': capital
                    })
            
            elif signal == -1 and position > 0:  # Sell signal
                # Sell all shares
                exit_price = price * (1 - self.slippage)
                proceeds = position * exit_price * (1 - self.commission)
                capital += proceeds
                
                trades.append({
                    'date': row.name,
                    'type': 'SELL',
                    'price': exit_price,
                    'shares': position,
                    'proceeds': proceeds,
                    'capital': capital,
                    'profit': proceeds - (position * entry_price * (1 + self.commission))
                })
                
                position = 0
                entry_price = 0
        
        # Close any open position at the end
        if position > 0:
            final_price = df.iloc[-1]['close'] * (1 - self.slippage)
            proceeds = position * final_price * (1 - self.commission)
            capital += proceeds
            
            trades.append({
                'date': df.index[-1],
                'type': 'SELL (Final)',
                'price': final_price,
                'shares': position,
                'proceeds': proceeds,
                'capital': capital,
                'profit': proceeds - (position * entry_price * (1 + self.commission))
            })
        
        # Add portfolio value to dataframe
        df['portfolio_value'] = portfolio_values
        df['position'] = positions_held
        
        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        portfolio_series = pd.Series(portfolio_values, index=df.index)
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_series)
        win_rate = self._calculate_win_rate(trades)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades,
            'df': df
        }
        
        self.results = results
        self.trades = trades
        
        return results
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize returns and volatility
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def _calculate_max_drawdown(self, portfolio_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_max = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative_max) / cumulative_max
        max_dd = drawdown.min()
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return winning_trades / len(trades)
    
    def plot_results(self, show_trades: bool = True) -> go.Figure:
        """
        Plot backtest results
        
        Args:
            show_trades: Whether to show buy/sell markers
            
        Returns:
            Plotly figure
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        df = self.results['df']
        trades = self.results['trades']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Value', 'Asset Price'),
            row_heights=[0.5, 0.5]
        )
        
        # Plot portfolio value
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[self.initial_capital] * len(df),
                name='Initial Capital',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        # Plot asset price
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close Price',
                line=dict(color='black', width=1)
            ),
            row=2, col=1
        )
        
        # Add buy/sell markers
        if show_trades and len(trades) > 0:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL (Final)']]
            
            if buy_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[t['date'] for t in buy_trades],
                        y=[t['price'] for t in buy_trades],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=12, color='green')
                    ),
                    row=2, col=1
                )
            
            if sell_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[t['date'] for t in sell_trades],
                        y=[t['price'] for t in sell_trades],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=12, color='red')
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results - Total Return: {self.results['total_return_pct']:.2f}%",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            yaxis2_title="Price ($)",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def print_summary(self):
        """Print backtest summary"""
        if self.results is None:
            print("No backtest results available.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital:     ${self.results['initial_capital']:,.2f}")
        print(f"Final Value:         ${self.results['final_value']:,.2f}")
        print(f"Total Return:        {self.results['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:        {self.results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {self.results['max_drawdown_pct']:.2f}%")
        print(f"Number of Trades:    {self.results['num_trades']}")
        print(f"Win Rate:            {self.results['win_rate']*100:.1f}%")
        print("="*60 + "\n")
