from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime

Base = declarative_base()

class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    strategy_type = Column(String, nullable=False) # e.g., 'RL_PPO', 'RL_A2C'
    balance = Column(Float, nullable=False, default=10000.0)
    score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    open_positions = relationship("OpenPosition", back_populates="agent")
    trade_history = relationship("TradeHistory", back_populates="agent")

class OpenPosition(Base):
    __tablename__ = 'open_positions'
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id'))
    symbol = Column(String, nullable=False)
    position_type = Column(String, nullable=False) # 'BUY' or 'SELL'
    lots = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    open_time = Column(DateTime, default=datetime.datetime.utcnow)
    margin_invested = Column(Float, nullable=False)
    brokerage_fee = Column(Float, nullable=False)

    agent = relationship("Agent", back_populates="open_positions")

class TradeHistory(Base):
    __tablename__ = 'trade_history'
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id'))
    symbol = Column(String, nullable=False)
    position_type = Column(String, nullable=False) # 'BUY' or 'SELL'
    lots = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, default=datetime.datetime.utcnow)
    margin_invested = Column(Float, nullable=False)
    brokerage_fee = Column(Float, nullable=False)
    gross_profit = Column(Float, nullable=False)
    net_profit = Column(Float, nullable=False) # gross_profit - brokerage_fee

    agent = relationship("Agent", back_populates="trade_history")

class SentimentRecord(Base):
    __tablename__ = 'sentiment_records'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False) # e.g., 'EURUSD=X' or 'GLOBAL'
    timestamp = Column(DateTime, nullable=False, index=True)
    score = Column(Float, nullable=False) # -1.0 to 1.0

def init_db(db_path='sqlite:///database.db'):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
