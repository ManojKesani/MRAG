import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelSession
from sqlalchemy import text

# Setup module-level logger
logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://mrag:mrag_secret@localhost:5432/mrag",
)

# engine: Set echo=True only for local debugging to see raw SQL
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

async def init_db() -> None:
    """Create all tables (idempotent)."""
    logger.info("Initializing database: Checking and creating tables...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Database initialization FAILED: {e}")
        raise

@asynccontextmanager
async def get_session() -> AsyncGenerator[SQLModelSession, None]:
    """Yield an async SQLModel session with transaction logging."""
    start_time = time.perf_counter()
    session_id = id(start_time) # Simple unique ID for tracing the session
    
    logger.debug(f"[DB SESSION {session_id}] Opening session.")
    
    async with SQLModelSession(engine) as session:
        try:
            yield session
            await session.commit()
            duration = time.perf_counter() - start_time
            logger.debug(f"[DB SESSION {session_id}] Committed | Duration: {duration:.3f}s")
        except Exception as e:
            await session.rollback()
            duration = time.perf_counter() - start_time
            logger.error(f"[DB SESSION {session_id}] ROLLBACK due to error: {e} | Duration: {duration:.3f}s")
            raise
        finally:
            # SQLModelSession closes automatically via 'async with', 
            # but we log the end of the context
            pass

async def check_db_connection() -> None:
    """Verify Postgres connection on startup."""
    logger.info("Checking database connection...")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Database is unreachable at {DATABASE_URL}. Error: {e}")
        raise