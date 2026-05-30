# Forex AI Arena - Context Guide for Claude

Welcome, Claude! This document provides context on the **Forex AI Arena** project so you can easily understand its architecture, rules, and how to extend it.

## Project Goal
This is a local gamified platform for training AI agents (specifically using Reinforcement Learning) to trade Forex. The agents compete using paper money under strict financial rules (leverage and brokerage fees). The ultimate goal is to find an optimal autonomous trading strategy.

## Tech Stack Overview
- **Language**: Python 3.10+
- **Data Source**: `yfinance` (fetching real-time and historical Forex data).
- **Reinforcement Learning**: `stable-baselines3` (currently using PPO algorithm).
- **Environment**: Custom `gymnasium` environment (`src/engine/env.py`) representing the trading market.
- **Database**: `SQLite` via `SQLAlchemy` (`src/db_models.py`). Used to store Agents, their balances, open positions, and historical trades.
- **Frontend UI**: `Streamlit` (`app.py`). Provides a real-time dashboard to monitor market data, the Agent Leaderboard, and open positions.
- **Hybrid AI Approach**: We plan to use local LLMs (via `Ollama`) to read news and economic calendars, converting text into a float "Sentiment Score" (-1.0 to 1.0) which is fed into the RL environment as an extra observation.

## Financial Math Rules (Strict)
The core logic resides in `src/engine/financial.py`. Please strictly adhere to these rules when modifying trading logic:
- **Standard Lot**: 100,000 units. The system defaults to trading *micro lots* (0.01 = 1,000 units).
- **Leverage**: Configured in `config.yaml` (default `1:100`). It dictates how much margin is required to open a position.
- **Brokerage Fee (Critical)**: We charge a `5% fee` on the **Invested Margin**, not the total position size. 
  - **Mechanic**: This fee is deducted *immediately* when an agent opens a position (Buy or Sell). Thus, every trade starts slightly negative.

## Project Structure
```text
forex-ai-arena/
├── config.yaml          # Global settings (pairs, leverage, fees)
├── app.py               # Streamlit Dashboard Entrypoint
├── train_agent.py       # Script to fetch data, init the env, and train a PPO Agent
├── database.db          # SQLite Database
├── src/
│   ├── data/            
│   │   └── fetcher.py   # yfinance wrappers
│   ├── engine/          
│   │   ├── env.py       # The Gymnasium RL Environment
│   │   └── financial.py # Margin, Fee, and PnL math
│   ├── ai/              
│   │   └── agents.py    # LLM Sentiment integration (Ollama)
│   └── db_models.py     # SQLAlchemy Schemas
```

## How to Run
- **To start the dashboard**: Run `streamlit run app.py` (Usually available at `http://localhost:8501`).
- **To train a new agent**: Run `python train_agent.py`. This will fetch recent data, train a new agent for a few thousand steps, and save it to the database.

## Extension Guidelines for Claude
- If adding new indicators (like RSI or MACD), add them to the data fetcher and append them to the `observation_space` in `src/engine/env.py`.
- If altering the database schema in `db_models.py`, remember that there is no Alembic migration setup yet; you might need to recreate `database.db`.
- When dealing with pandas DataFrames inside the `ForexEnv` step/observation methods, always cast the numpy values explicitly to `float()` to avoid `stable-baselines3` array shape exceptions.
