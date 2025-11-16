"""
Main data fetcher that orchestrates data collection from multiple sources
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .api_clients import YFinanceClient, AlphaVantageClient, FinnhubClient
from database import (
    get_engine, get_session, StockPrice, StockInfo,
    StockPriceRepository, StockInfoRepository, APICallLogRepository
)
from config import config
from utils.logger import logger
from utils.helpers import validate_symbol, validate_date_range


class StockDataFetcher:
    """
    Main class for fetching stock data from multiple sources
    Handles data aggregation, deduplication, and storage
    """
    
    def __init__(self, database_url: str = None):
        """Initialize data fetcher with API clients and database connection"""
        # Initialize API clients
        self.yfinance_client = YFinanceClient()
        
        try:
            self.alphavantage_client = AlphaVantageClient()
        except ValueError as e:
            logger.warning(f"Alpha Vantage client not initialized: {e}")
            self.alphavantage_client = None
        
        try:
            self.finnhub_client = FinnhubClient()
        except ValueError as e:
            logger.warning(f"Finnhub client not initialized: {e}")
            self.finnhub_client = None
        
        # Initialize database
        db_url = database_url or config.DATABASE_URL
        self.engine = get_engine(db_url)
        self.session = get_session(self.engine)
        
        # Initialize repositories
        self.price_repo = StockPriceRepository(self.session)
        self.info_repo = StockInfoRepository(self.session)
        self.api_log_repo = APICallLogRepository(self.session)
        
        logger.info("StockDataFetcher initialized successfully")
    
    def fetch_historical_data(self, symbol: str, start_date: str = None,
                             end_date: str = None, interval: str = '1d',
                             source: str = 'yfinance') -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            source: Data source ('yfinance' or 'alphavantage')
            
        Returns:
            DataFrame with historical data
        """
        symbol = validate_symbol(symbol)
        logger.info(f"Fetching historical data for {symbol} from {source}")
        
        try:
            if source == 'yfinance':
                # Calculate period from dates
                if start_date:
                    start = pd.to_datetime(start_date)
                    end = pd.to_datetime(end_date) if end_date else datetime.now()
                    days_diff = (end - start).days
                    
                    if days_diff <= 7:
                        period = '7d'
                    elif days_diff <= 30:
                        period = '1mo'
                    elif days_diff <= 90:
                        period = '3mo'
                    elif days_diff <= 180:
                        period = '6mo'
                    elif days_diff <= 365:
                        period = '1y'
                    else:
                        period = 'max'
                else:
                    period = '1y'  # Default to 1 year
                
                df = self.yfinance_client.get_stock_data(symbol, period=period, interval=interval)
                self.api_log_repo.log_call('yfinance', f'/history/{symbol}', success=True)
                
            elif source == 'alphavantage':
                if not self.alphavantage_client:
                    raise ValueError("Alpha Vantage client not available")
                
                if interval in ['1d', '1wk', '1mo']:
                    df = self.alphavantage_client.get_daily_data(symbol, outputsize='full')
                else:
                    df = self.alphavantage_client.get_intraday_data(symbol, interval=interval)
                
                self.api_log_repo.log_call('alphavantage', f'/timeseries/{symbol}', success=True)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                # Store data in database
                self._store_price_data(symbol, df, interval)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            self.api_log_repo.log_call(source, f'/history/{symbol}', success=False, error_message=str(e))
            raise
    
    def fetch_realtime_quote(self, symbol: str, source: str = 'yfinance') -> Dict:
        """
        Fetch real-time quote for a symbol
        
        Args:
            symbol: Stock ticker symbol
            source: Data source ('yfinance', 'alphavantage', or 'finnhub')
            
        Returns:
            Dictionary with current quote data
        """
        symbol = validate_symbol(symbol)
        logger.info(f"Fetching real-time quote for {symbol} from {source}")
        
        try:
            if source == 'yfinance':
                quote = self.yfinance_client.get_realtime_quote(symbol)
                self.api_log_repo.log_call('yfinance', f'/quote/{symbol}', success=True)
                
            elif source == 'alphavantage':
                if not self.alphavantage_client:
                    raise ValueError("Alpha Vantage client not available")
                quote = self.alphavantage_client.get_quote(symbol)
                self.api_log_repo.log_call('alphavantage', f'/quote/{symbol}', success=True)
                
            elif source == 'finnhub':
                if not self.finnhub_client:
                    raise ValueError("Finnhub client not available")
                quote = self.finnhub_client.get_quote(symbol)
                self.api_log_repo.log_call('finnhub', f'/quote/{symbol}', success=True)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            logger.info(f"Successfully fetched quote for {symbol}: ${quote.get('price', 'N/A')}")
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            self.api_log_repo.log_call(source, f'/quote/{symbol}', success=False, error_message=str(e))
            return {}
    
    def fetch_company_info(self, symbol: str, source: str = 'yfinance') -> Dict:
        """
        Fetch company information
        
        Args:
            symbol: Stock ticker symbol
            source: Data source ('yfinance' or 'finnhub')
            
        Returns:
            Dictionary with company information
        """
        symbol = validate_symbol(symbol)
        logger.info(f"Fetching company info for {symbol} from {source}")
        
        try:
            if source == 'yfinance':
                info = self.yfinance_client.get_stock_info(symbol)
                self.api_log_repo.log_call('yfinance', f'/info/{symbol}', success=True)
                
            elif source == 'finnhub':
                if not self.finnhub_client:
                    raise ValueError("Finnhub client not available")
                info = self.finnhub_client.get_company_profile(symbol)
                self.api_log_repo.log_call('finnhub', f'/profile/{symbol}', success=True)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            # Store company info in database
            if info:
                self._store_company_info(info)
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            self.api_log_repo.log_call(source, f'/info/{symbol}', success=False, error_message=str(e))
            return {}
    
    def fetch_multiple_quotes(self, symbols: List[str], source: str = 'yfinance') -> Dict[str, Dict]:
        """
        Fetch quotes for multiple symbols
        
        Args:
            symbols: List of stock ticker symbols
            source: Data source
            
        Returns:
            Dictionary mapping symbols to their quotes
        """
        quotes = {}
        for symbol in symbols:
            try:
                quote = self.fetch_realtime_quote(symbol, source=source)
                if quote:
                    quotes[symbol] = quote
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")
                continue
        
        return quotes
    
    def _store_price_data(self, symbol: str, df: pd.DataFrame, interval: str):
        """Store price data in database"""
        try:
            stock_prices = []
            for timestamp, row in df.iterrows():
                # Check if record already exists
                if not self.price_repo.exists(symbol, timestamp, interval):
                    stock_price = StockPrice(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        interval=interval
                    )
                    stock_prices.append(stock_price)
            
            if stock_prices:
                count = self.price_repo.bulk_create(stock_prices)
                logger.info(f"Stored {count} new price records for {symbol}")
            else:
                logger.info(f"No new price records to store for {symbol}")
                
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
    
    def _store_company_info(self, info: Dict):
        """Store company information in database"""
        try:
            stock_info = StockInfo(
                symbol=info.get('symbol'),
                company_name=info.get('company_name'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('market_cap'),
                pe_ratio=info.get('pe_ratio'),
                dividend_yield=info.get('dividend_yield'),
                week_52_high=info.get('week_52_high'),
                week_52_low=info.get('week_52_low'),
                description=info.get('description')
            )
            
            self.info_repo.create_or_update(stock_info)
            logger.info(f"Stored company info for {info.get('symbol')}")
            
        except Exception as e:
            logger.error(f"Error storing company info: {e}")
    
    def get_stored_data(self, symbol: str, start_date: str = None,
                       end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        """
        Retrieve stored price data from database
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with stored data
        """
        symbol = validate_symbol(symbol)
        
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None
        
        prices = self.price_repo.get_by_symbol(symbol, start, end, interval)
        
        if not prices:
            logger.info(f"No stored data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = [{
            'timestamp': p.timestamp,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume
        } for p in prices]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Retrieved {len(df)} stored records for {symbol}")
        return df
    
    def close(self):
        """Close database session"""
        self.session.close()
        logger.info("StockDataFetcher closed")
