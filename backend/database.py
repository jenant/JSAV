from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jsav.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    entry_date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    overall_sentiment = Column(Float, default=0.0)  # -1 to 1
    overall_mood = Column(String, nullable=True)

    # Life category scores (-1 to 1)
    family = Column(Float, nullable=True)
    relationships = Column(Float, nullable=True)
    finances = Column(Float, nullable=True)
    mental = Column(Float, nullable=True)
    hobbies = Column(Float, nullable=True)
    academics_career = Column(Float, nullable=True)

    # Raw analysis from Claude
    analysis_notes = Column(Text, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
