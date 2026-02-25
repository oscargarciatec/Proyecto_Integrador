from sqlalchemy import Column, String, TIMESTAMP, Integer, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import BYTEA
from .database import Base

class HubConversation(Base):
    __tablename__ = "hub_conversation"
    kh_conversation = Column(BYTEA, primary_key=True)
    ax_conversation = Column(String)

class SatHistoricalChats(Base):
    __tablename__ = "sat_historical_chats"
    kh_user_agent_conversation = Column(BYTEA, primary_key=True)
    ct_valid_from_dt = Column(TIMESTAMP, primary_key=True)
    ab_feedback = Column(Boolean)
    ax_message_type = Column(String)