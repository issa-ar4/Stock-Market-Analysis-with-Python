"""
Unit tests for data ingestion module
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_ingestion import YFinanceClient, StockDataFetcher
from utils.helpers import validate_symbol


class TestYFinanceClient:
    """Test YFinance API client"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = YFinanceClient()
        self.test_symbol = 'AAPL'
    
    def test_get_stock_data(self):
        """Test fetching stock data"""
        df = self.client.get_stock_data(self.test_symbol, period='5d', interval='1d')
        
        assert not df.empty
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_get_stock_info(self):
        """Test fetching company info"""
        info = self.client.get_stock_info(self.test_symbol)
        
        assert info is not None
        assert 'symbol' in info
        assert 'company_name' in info
        assert info['symbol'] == self.test_symbol
    
    def test_get_realtime_quote(self):
        """Test fetching real-time quote"""
        quote = self.client.get_realtime_quote(self.test_symbol)
        
        assert quote is not None
        assert 'symbol' in quote
        assert 'price' in quote
        assert quote['symbol'] == self.test_symbol


class TestHelpers:
    """Test helper functions"""
    
    def test_validate_symbol(self):
        """Test symbol validation"""
        assert validate_symbol('aapl') == 'AAPL'
        assert validate_symbol(' MSFT ') == 'MSFT'
        
        with pytest.raises(ValueError):
            validate_symbol('')
        
        with pytest.raises(ValueError):
            validate_symbol(None)


class TestStockDataFetcher:
    """Test main data fetcher"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fetcher = StockDataFetcher(database_url='sqlite:///:memory:')
        self.test_symbol = 'AAPL'
    
    def teardown_method(self):
        """Cleanup after tests"""
        self.fetcher.close()
    
    def test_fetch_historical_data(self):
        """Test fetching and storing historical data"""
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        df = self.fetcher.fetch_historical_data(
            symbol=self.test_symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        assert not df.empty
        assert len(df) > 0
    
    def test_fetch_realtime_quote(self):
        """Test fetching real-time quote"""
        quote = self.fetcher.fetch_realtime_quote(self.test_symbol)
        
        assert quote is not None
        assert 'symbol' in quote
        assert 'price' in quote
    
    def test_fetch_company_info(self):
        """Test fetching company information"""
        info = self.fetcher.fetch_company_info(self.test_symbol)
        
        assert info is not None
        assert 'symbol' in info
        assert 'company_name' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
