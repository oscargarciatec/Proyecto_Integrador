from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Reemplaza con tus credenciales locales de AlloyDB
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:proyecto_integrador@localhost:5432/spin-voyager"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Definimos el esquema para que SQLAlchemy sepa dónde buscar las tablas
metadata = MetaData(schema="multiagent_rag_model")
Base = declarative_base(metadata=metadata)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()