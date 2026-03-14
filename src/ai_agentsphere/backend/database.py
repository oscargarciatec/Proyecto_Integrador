from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
import os

# Reemplaza con tus credenciales locales de AlloyDB
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL", "postgresql://postgres:proyecto_integrador@localhost:5432/spin-voyager")

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"statement_cache_size": 0}
)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Definimos el esquema para que SQLAlchemy sepa dónde buscar las tablas
metadata = MetaData(schema="multiagent_rag_model")
Base = declarative_base(metadata=metadata)

async def get_db():
    async with SessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()