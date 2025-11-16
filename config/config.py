"""
Configuration management for the Stock Market Analysis Platform
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration"""
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///stock_data.db')
    
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Application
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Trading
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Data Settings
    DATA_REFRESH_INTERVAL = int(os.getenv('DATA_REFRESH_INTERVAL', 60))
    MAX_API_CALLS_PER_MINUTE = int(os.getenv('MAX_API_CALLS_PER_MINUTE', 5))
    
    # API Endpoints
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.ALPHA_VANTAGE_API_KEY:
            print("Warning: ALPHA_VANTAGE_API_KEY not set")
        if not cls.FINNHUB_API_KEY:
            print("Warning: FINNHUB_API_KEY not set")
        return True


# Create config instance
config = Config()
