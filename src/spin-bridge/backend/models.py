from sqlalchemy import Column, String, TIMESTAMP, Integer, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import BYTEA
from database import Base

class HubConversation(Base):
    __tablename__ = "hub_conversation"
    kh_conversation = Column(BYTEA, primary_key=True)
    ax_conversation = Column(String)

class SatHistoricalChats(Base):
    __tablename__ = "sat_compass_historical_chats"
    kh_user_agent_conversation = Column(BYTEA, primary_key=True)
    ct_valid_from_dt = Column(TIMESTAMP, primary_key=True)
    ab_feedback = Column(Boolean)
    ax_message_type = Column(String)
    ax_content = Column(String)
    aj_attachments = Column(JSON)

class LnkUserAgentsConversation(Base):
    __tablename__ = "lnk_users_agents_conversation"
    kh_user_agent_conversation = Column(BYTEA, primary_key=True)
    kh_user = Column(BYTEA)

class HubUser(Base):
    __tablename__ = "hub_user"
    kh_user = Column(BYTEA, primary_key=True)
    ax_user = Column(String)

class SatUsersData(Base):
    __tablename__ = "sat_compass_users_data"
    kh_user = Column(BYTEA, primary_key=True)
    ax_display_nm = Column(String)
    ax_email = Column(String)
    ai_current_flag = Column(Integer)
    