"""
Repository pattern for database operations
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from datetime import datetime, timedelta
from typing import List, Optional
from .models import StockPrice, StockInfo, APICallLog
from utils.logger import logger


class StockPriceRepository:
    """Repository for StockPrice operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, stock_price: StockPrice) -> StockPrice:
        """Create a new stock price record"""
        try:
            self.session.add(stock_price)
            self.session.commit()
            self.session.refresh(stock_price)
            return stock_price
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating stock price: {e}")
            raise
    
    def bulk_create(self, stock_prices: List[StockPrice]) -> int:
        """Bulk insert stock prices"""
        try:
            self.session.bulk_save_objects(stock_prices)
            self.session.commit()
            return len(stock_prices)
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error bulk creating stock prices: {e}")
            raise
    
    def get_by_symbol(self, symbol: str, start_date: datetime = None, 
                     end_date: datetime = None, interval: str = '1d') -> List[StockPrice]:
        """Get stock prices for a symbol within date range"""
        query = self.session.query(StockPrice).filter(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.interval == interval
            )
        )
        
        if start_date:
            query = query.filter(StockPrice.timestamp >= start_date)
        if end_date:
            query = query.filter(StockPrice.timestamp <= end_date)
        
        return query.order_by(StockPrice.timestamp).all()
    
    def get_latest(self, symbol: str, interval: str = '1d') -> Optional[StockPrice]:
        """Get the most recent price for a symbol"""
        return self.session.query(StockPrice).filter(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.interval == interval
            )
        ).order_by(desc(StockPrice.timestamp)).first()
    
    def exists(self, symbol: str, timestamp: datetime, interval: str) -> bool:
        """Check if a price record exists"""
        return self.session.query(StockPrice).filter(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.timestamp == timestamp,
                StockPrice.interval == interval
            )
        ).first() is not None


class StockInfoRepository:
    """Repository for StockInfo operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_or_update(self, stock_info: StockInfo) -> StockInfo:
        """Create or update stock info"""
        try:
            existing = self.session.query(StockInfo).filter(
                StockInfo.symbol == stock_info.symbol
            ).first()
            
            if existing:
                # Update existing record
                for key, value in stock_info.__dict__.items():
                    if not key.startswith('_') and key != 'id':
                        setattr(existing, key, value)
                existing.last_updated = datetime.utcnow()
                self.session.commit()
                return existing
            else:
                # Create new record
                self.session.add(stock_info)
                self.session.commit()
                self.session.refresh(stock_info)
                return stock_info
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating/updating stock info: {e}")
            raise
    
    def get_by_symbol(self, symbol: str) -> Optional[StockInfo]:
        """Get stock info by symbol"""
        return self.session.query(StockInfo).filter(
            StockInfo.symbol == symbol
        ).first()
    
    def get_all(self) -> List[StockInfo]:
        """Get all stock info records"""
        return self.session.query(StockInfo).all()


class APICallLogRepository:
    """Repository for API call logging"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def log_call(self, api_name: str, endpoint: str = None, 
                success: bool = True, error_message: str = None):
        """Log an API call"""
        try:
            log = APICallLog(
                api_name=api_name,
                endpoint=endpoint,
                success=1 if success else 0,
                error_message=error_message
            )
            self.session.add(log)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error logging API call: {e}")
    
    def get_calls_in_last_minute(self, api_name: str) -> int:
        """Get number of API calls in the last minute"""
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        count = self.session.query(APICallLog).filter(
            and_(
                APICallLog.api_name == api_name,
                APICallLog.timestamp >= one_minute_ago
            )
        ).count()
        return count
    
    def can_make_call(self, api_name: str, max_calls_per_minute: int) -> bool:
        """Check if we can make an API call without exceeding rate limit"""
        recent_calls = self.get_calls_in_last_minute(api_name)
        return recent_calls < max_calls_per_minute
