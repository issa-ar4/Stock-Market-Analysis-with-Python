"""
Database models for storing stock market data
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class StockPrice(Base):
    """Model for storing historical and real-time stock prices"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    interval = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_timestamp_interval', 'symbol', 'timestamp', 'interval'),
    )
    
    def __repr__(self):
        return f"<StockPrice(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'interval': self.interval
        }


class StockInfo(Base):
    """Model for storing stock metadata and company information"""
    __tablename__ = 'stock_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    company_name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    dividend_yield = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    description = Column(String(1000))
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<StockInfo(symbol='{self.symbol}', company_name='{self.company_name}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'dividend_yield': self.dividend_yield,
            'week_52_high': self.week_52_high,
            'week_52_low': self.week_52_low,
            'description': self.description,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class APICallLog(Base):
    """Model for tracking API calls to manage rate limits"""
    __tablename__ = 'api_call_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_name = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    success = Column(Integer, default=1)  # 1 for success, 0 for failure
    error_message = Column(String(500))
    
    def __repr__(self):
        return f"<APICallLog(api_name='{self.api_name}', timestamp='{self.timestamp}')>"


# Database utility functions
def get_engine(database_url: str):
    """Create database engine"""
    return create_engine(database_url, echo=False)


def get_session(engine):
    """Create database session"""
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(database_url: str):
    """Initialize database - create all tables"""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
