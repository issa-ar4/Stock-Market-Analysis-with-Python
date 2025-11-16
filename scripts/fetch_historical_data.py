#!/usr/bin/env python
"""
Script to fetch and store historical stock data
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion import StockDataFetcher
from utils.logger import logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fetch and store historical stock data'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)'
    )
    
    parser.add_argument(
        '--period', '-p',
        type=str,
        default='1y',
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
        help='Time period to fetch (default: 1y)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default='1d',
        choices=['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'],
        help='Data interval (default: 1d)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='yfinance',
        choices=['yfinance', 'alphavantage'],
        help='Data source (default: yfinance)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Also fetch company information'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    try:
        logger.info(f"Starting data fetch for {args.symbol}")
        logger.info(f"Period: {args.period}, Interval: {args.interval}, Source: {args.source}")
        
        # Initialize data fetcher
        fetcher = StockDataFetcher()
        
        # Calculate date range based on period
        end_date = datetime.now()
        period_map = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            'max': 7300  # ~20 years
        }
        
        days = period_map.get(args.period, 365)
        start_date = end_date - timedelta(days=days)
        
        # Fetch historical data
        df = fetcher.fetch_historical_data(
            symbol=args.symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval=args.interval,
            source=args.source
        )
        
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} records for {args.symbol}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"\nSample data (last 5 records):")
            print(df.tail())
        else:
            logger.warning(f"No data fetched for {args.symbol}")
        
        # Fetch company info if requested
        if args.info:
            logger.info(f"\nFetching company information for {args.symbol}...")
            info = fetcher.fetch_company_info(args.symbol, source=args.source)
            
            if info:
                logger.info("\nCompany Information:")
                logger.info(f"  Name: {info.get('company_name')}")
                logger.info(f"  Sector: {info.get('sector')}")
                logger.info(f"  Industry: {info.get('industry')}")
                logger.info(f"  Market Cap: {info.get('market_cap')}")
        
        # Close fetcher
        fetcher.close()
        
        logger.info("\nData fetch completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during data fetch: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
