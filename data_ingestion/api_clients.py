"""
API clients for various stock data providers
"""
import time
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from config import config
from utils.logger import logger
from utils.helpers import retry_on_failure


class YFinanceClient:
    """Client for Yahoo Finance API using yfinance library"""
    
    def __init__(self):
        self.name = "YFinance"
        logger.info(f"Initialized {self.name} client")
    
    @retry_on_failure(max_retries=3, delay=2)
    def get_stock_data(self, symbol: str, period: str = '1mo', 
                       interval: str = '1d') -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {symbol} data from {self.name} (period={period}, interval={interval})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {self.name}: {e}")
            raise
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get company information
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            logger.info(f"Fetching company info for {symbol} from {self.name}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant fields
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'description': info.get('longBusinessSummary')
            }
            
            logger.info(f"Successfully fetched info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info from {self.name}: {e}")
            raise
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with current price data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            quote = {
                'symbol': symbol,
                'price': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'timestamp': latest.name
            }
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote: {e}")
            return {}


class AlphaVantageClient:
    """Client for Alpha Vantage API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.name = "AlphaVantage"
        self.base_url = config.ALPHA_VANTAGE_BASE_URL
        logger.info(f"Initialized {self.name} client")
    
    @retry_on_failure(max_retries=3, delay=2)
    def get_intraday_data(self, symbol: str, interval: str = '5min') -> pd.DataFrame:
        """
        Get intraday stock data
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            
        Returns:
            DataFrame with intraday data
        """
        try:
            logger.info(f"Fetching intraday data for {symbol} from {self.name}")
            data, meta_data = self.ts.get_intraday(
                symbol=symbol,
                interval=interval,
                outputsize='full'
            )
            
            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            logger.info(f"Successfully fetched {len(data)} intraday records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data from {self.name}: {e}")
            raise
    
    @retry_on_failure(max_retries=3, delay=2)
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get daily stock data
        
        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (last 100 data points) or 'full' (20+ years)
            
        Returns:
            DataFrame with daily data
        """
        try:
            logger.info(f"Fetching daily data for {symbol} from {self.name}")
            data, meta_data = self.ts.get_daily(
                symbol=symbol,
                outputsize=outputsize
            )
            
            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            logger.info(f"Successfully fetched {len(data)} daily records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching daily data from {self.name}: {e}")
            raise
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            url = self.base_url
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' not in data:
                logger.warning(f"No quote data returned for {symbol}")
                return {}
            
            quote = data['Global Quote']
            return {
                'symbol': quote.get('01. symbol'),
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': quote.get('07. latest trading day')
            }
            
        except Exception as e:
            logger.error(f"Error fetching quote from {self.name}: {e}")
            return {}


class FinnhubClient:
    """Client for Finnhub API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.FINNHUB_API_KEY
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        self.base_url = config.FINNHUB_BASE_URL
        self.name = "Finnhub"
        logger.info(f"Initialized {self.name} client")
    
    @retry_on_failure(max_retries=3, delay=2)
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': symbol,
                'price': data.get('c'),  # current price
                'high': data.get('h'),   # high price of the day
                'low': data.get('l'),    # low price of the day
                'open': data.get('o'),   # open price of the day
                'previous_close': data.get('pc'),
                'timestamp': datetime.fromtimestamp(data.get('t', 0))
            }
            
        except Exception as e:
            logger.error(f"Error fetching quote from {self.name}: {e}")
            return {}
    
    @retry_on_failure(max_retries=3, delay=2)
    def get_company_profile(self, symbol: str) -> Dict:
        """
        Get company profile
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            url = f"{self.base_url}/stock/profile2"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': symbol,
                'company_name': data.get('name'),
                'industry': data.get('finnhubIndustry'),
                'market_cap': data.get('marketCapitalization'),
                'country': data.get('country'),
                'currency': data.get('currency'),
                'exchange': data.get('exchange'),
                'ipo': data.get('ipo'),
                'logo': data.get('logo'),
                'phone': data.get('phone'),
                'weburl': data.get('weburl')
            }
            
        except Exception as e:
            logger.error(f"Error fetching company profile from {self.name}: {e}")
            return {}
