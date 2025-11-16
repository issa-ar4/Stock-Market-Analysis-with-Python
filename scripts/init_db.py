#!/usr/bin/env python
"""
Script to initialize the database and create tables
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import init_db
from config import config
from utils.logger import logger


def main():
    """Initialize database"""
    try:
        logger.info("Initializing database...")
        logger.info(f"Database URL: {config.DATABASE_URL}")
        
        engine = init_db(config.DATABASE_URL)
        
        logger.info("Database initialized successfully!")
        logger.info("Tables created:")
        logger.info("  - stock_prices")
        logger.info("  - stock_info")
        logger.info("  - api_call_logs")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
