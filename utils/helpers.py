"""
Helper utilities for the Stock Market Analysis Platform
"""
from datetime import datetime, timedelta
import pandas as pd
from typing import Union, List


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Normalized symbol (uppercase, stripped)
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    return symbol.strip().upper()


def validate_date_range(start_date: Union[str, datetime], 
                        end_date: Union[str, datetime] = None) -> tuple:
    """
    Validate and normalize date range
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime), defaults to today
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")
    
    return start_date, end_date


def parse_interval(interval: str) -> str:
    """
    Parse and validate time interval
    
    Args:
        interval: Time interval (e.g., '1d', '1h', '5m')
        
    Returns:
        Validated interval string
    """
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
    
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Must be one of {valid_intervals}")
    
    return interval


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    
    return ((new_value - old_value) / old_value) * 100


def format_currency(value: float, symbol: str = "$") -> str:
    """
    Format value as currency
    
    Args:
        value: Numeric value
        symbol: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value
        
    Returns:
        Formatted percentage string
    """
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_failure(max_retries: int = 3, delay: int = 1):
    """
    Decorator to retry a function on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator
