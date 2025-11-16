"""
Alpaca API Client for Paper/Live Trading
"""
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from config import config


class AlpacaClient:
    """
    Client for interacting with Alpaca Trading API
    Supports both paper and live trading
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca client
        
        Args:
            paper_trading: If True, use paper trading account
        """
        self.api_key = config.ALPACA_API_KEY
        self.secret_key = config.ALPACA_SECRET_KEY
        self.paper_trading = paper_trading
        
        # Set base URL based on trading mode
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
    
    def get_account(self) -> Dict:
        """Get account information"""
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol"""
        try:
            url = f"{self.base_url}/v2/positions/{symbol}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_orders(self, status: str = 'all') -> List[Dict]:
        """
        Get orders
        
        Args:
            status: Order status filter ('all', 'open', 'closed', 'canceled')
        """
        url = f"{self.base_url}/v2/orders"
        params = {'status': status}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Place an order
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
        """
        url = f"{self.base_url}/v2/orders"
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force
        }
        
        if limit_price is not None:
            data['limit_price'] = limit_price
        if stop_price is not None:
            data['stop_price'] = stop_price
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order"""
        url = f"{self.base_url}/v2/orders/{order_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
    
    def cancel_all_orders(self) -> List[Dict]:
        """Cancel all open orders"""
        url = f"{self.base_url}/v2/orders"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def close_position(self, symbol: str) -> Dict:
        """Close a position"""
        url = f"{self.base_url}/v2/positions/{symbol}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def close_all_positions(self) -> List[Dict]:
        """Close all positions"""
        url = f"{self.base_url}/v2/positions"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical price bars
        
        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start datetime
            end: End datetime
            limit: Max number of bars to return
        """
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        
        params = {
            'timeframe': timeframe,
            'limit': limit
        }
        
        if start:
            params['start'] = start.isoformat()
        if end:
            params['end'] = end.isoformat()
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json().get('bars', [])
    
    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest quote for a symbol"""
        url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get('quote', {})
    
    def get_latest_trade(self, symbol: str) -> Dict:
        """Get latest trade for a symbol"""
        url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get('trade', {})
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        url = f"{self.base_url}/v2/clock"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        clock = response.json()
        return clock.get('is_open', False)
    
    def get_calendar(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]:
        """Get market calendar"""
        url = f"{self.base_url}/v2/calendar"
        params = {}
        
        if start:
            params['start'] = start.strftime('%Y-%m-%d')
        if end:
            params['end'] = end.strftime('%Y-%m-%d')
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
