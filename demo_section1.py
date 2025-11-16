#!/usr/bin/env python
"""
Demo script showcasing Section 1: Real-Time Data Ingestion
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion import StockDataFetcher
from utils.logger import logger
from utils.helpers import format_currency, format_percentage, calculate_percentage_change
from datetime import datetime, timedelta


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_realtime_quotes():
    """Demonstrate real-time quote fetching"""
    print_header("Real-Time Stock Quotes")
    
    fetcher = StockDataFetcher()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print(f"Fetching real-time quotes for: {', '.join(symbols)}\n")
    
    for symbol in symbols:
        try:
            quote = fetcher.fetch_realtime_quote(symbol, source='yfinance')
            
            if quote:
                print(f"{symbol:6} | Price: {format_currency(quote.get('price', 0)):>12} | "
                      f"Volume: {quote.get('volume', 0):>12,}")
        except Exception as e:
            print(f"{symbol:6} | Error: {str(e)}")
    
    fetcher.close()


def demo_historical_data():
    """Demonstrate historical data fetching"""
    print_header("Historical Data Fetching")
    
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Fetching 30 days of historical data for {symbol}...\n")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            source='yfinance'
        )
        
        if not df.empty:
            print(f"Retrieved {len(df)} records")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"\nLast 5 trading days:\n")
            print(df.tail())
            
            # Calculate statistics
            latest_close = df['close'].iloc[-1]
            first_close = df['close'].iloc[0]
            pct_change = calculate_percentage_change(first_close, latest_close)
            
            print(f"\n30-Day Performance:")
            print(f"  Starting Price: {format_currency(first_close)}")
            print(f"  Current Price:  {format_currency(latest_close)}")
            print(f"  Change:         {format_percentage(pct_change)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    fetcher.close()


def demo_company_info():
    """Demonstrate company information fetching"""
    print_header("Company Information")
    
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Fetching company information for {symbol}...\n")
    
    try:
        info = fetcher.fetch_company_info(symbol, source='yfinance')
        
        if info:
            print(f"Company Name: {info.get('company_name')}")
            print(f"Sector:       {info.get('sector')}")
            print(f"Industry:     {info.get('industry')}")
            
            if info.get('market_cap'):
                print(f"Market Cap:   ${info.get('market_cap'):,}")
            
            if info.get('pe_ratio'):
                print(f"P/E Ratio:    {info.get('pe_ratio'):.2f}")
            
            if info.get('dividend_yield'):
                print(f"Div. Yield:   {info.get('dividend_yield')*100:.2f}%")
            
            if info.get('week_52_high') and info.get('week_52_low'):
                print(f"52W Range:    {format_currency(info.get('week_52_low'))} - "
                      f"{format_currency(info.get('week_52_high'))}")
            
            if info.get('description'):
                desc = info.get('description')
                print(f"\nDescription:")
                print(f"  {desc[:200]}...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    fetcher.close()


def demo_data_retrieval():
    """Demonstrate retrieving stored data from database"""
    print_header("Retrieving Stored Data")
    
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    
    print(f"Retrieving stored data for {symbol} from database...\n")
    
    try:
        # First, ensure we have some data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Fetch and store
        fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        # Now retrieve from database
        df = fetcher.get_stored_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if not df.empty:
            print(f"Retrieved {len(df)} records from database")
            print(f"\nStored data:\n")
            print(df)
        else:
            print("No data found in database")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    fetcher.close()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  SECTION 1: REAL-TIME DATA INGESTION DEMO")
    print("  Stock Market Analysis Platform")
    print("=" * 70)
    
    try:
        # Run demos
        demo_realtime_quotes()
        demo_historical_data()
        demo_company_info()
        demo_data_retrieval()
        
        print("\n" + "=" * 70)
        print("  Demo completed successfully!")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
