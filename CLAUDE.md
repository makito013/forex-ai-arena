# Forex AI Arena - Context Guide for Claude

Welcome, Claude! This document provides context on the **Forex AI Arena** project so you can easily understand its architecture, rules, and how to extend it.

## Project Goal
A generational AI training platform for Forex trading. Multiple RL agents compete simultaneously on paper money under real broker rules (leverage, spread, swap). Each generation eliminates the worst performers (extracting why they failed), promotes the best, and the next generation inherits combined wisdom. Goal: evolve an agent capable of autonomous, profitable paper trading — and eventually deploy it in a real-time monitoring interface.

## Tech Stack Overview
- **Language**: Python 3.10+
- **Data Source**: `yfinance` (fetching real-time and historical Forex data).
- **Reinforcement Learning**: `stable-baselines3` (currently using PPO algorithm).
- **Environment**: Custom `gymnasium` environment (`src/engine/env.py`) representing the trading market.
- **Database**: `SQLite` via `SQLAlchemy` (`src/db_models.py`). Used to store Agents, their balances, open positions, and historical trades.
- **Frontend UI**: `Streamlit` (`app.py`). Provides a real-time dashboard to monitor market data, the Agent Leaderboard, and open positions.
- **Sentiment AI**: Claude API (Haiku 4.5) replaces Ollama for news sentiment scoring. See `docs/NEWS_DATA_GUIDE.md` for news data sources (historical and real-time).
- **Generational Evolution**: Multiple agents compete per generation; losers are analyzed by Claude API (failure report), winners seed the next generation. See `src/evolution/` (planned).

## Financial Math Rules (Strict)
The core logic resides in `src/engine/financial.py`. Please strictly adhere to these rules when modifying trading logic:
- **Standard Lot**: 100,000 units. The system defaults to trading *micro lots* (0.01 = 1,000 units).
  - **Per-asset contract sizes**: Non-forex assets override this via `trading.contract_sizes` in `config.yaml` (e.g. Gold/XAUUSD = 100 oz per lot). `FinancialEngine.get_contract_size(symbol)` resolves it by substring match.
- **Leverage**: Configured in `config.yaml` (default `1:100`). It dictates how much margin is required to open a position.
- **Trading Costs (Critical)**: Two models, switched by `costs.enabled` in `config.yaml`:
  - **Realistic (default, `enabled: true`)**: real-broker costs — **spread** (paid round-trip via bid/ask, per-pair in `costs.spreads_pips`), **commission** (`commission_per_lot`, round-turn USD), and **swap** (`swap_per_lot_per_night`, charged per night held). Implemented in `FinancialEngine.transaction_cost()` / `swap_cost()`. Use this for any test meant to reflect reality.
  - **Legacy (`enabled: false`)**: the old fictional `5% fee` on the **Invested Margin**, deducted when opening a position. Kept only for backward comparison and the Math Test tab.
- **Risk Management (enforced by `ForexEnv`, visible to the agent via observations 27-29)**:
  - **Margin check**: An order is rejected if balance can't cover margin + fee, or if the required margin exceeds `max_margin_usage_pct` of equity (no all-in).
  - **Stop-Out**: The broker force-closes the position when equity / margin falls below `stop_out_level`.
  - **Negative Balance Protection**: Balance floors at 0 after a liquidation.
  - **Bankruptcy**: The episode terminates when equity drops below 5% of the initial balance.
  - **Max Drawdown**: Tracked per env (`env.max_drawdown`, worst peak-to-valley equity loss, persists across resets). Goal-Oriented winners must stay under the configured drawdown limit.

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

## Multi-Agent Decision Protocol

**IMPORTANT: Before implementing any significant feature, architectural change, or new subsystem, you MUST run a multi-agent debate.** Do not write code until all agents have weighed in and a consolidated decision has been reached.

### When to trigger the debate
- Any change to `env.py` (observation space, reward function, action space)
- Any new ML model or training architecture
- Any new data source, indicator, or feature
- Any change to the financial math in `financial.py`
- Any new integration (news feeds, external APIs, Ollama models)
- Any refactor that touches more than 2 files

### The 5 Specialists

| Agent | File | Responsibilities |
|-------|------|-----------------|
| `forex-specialist` | `.claude/agents/forex-specialist.md` | Trading strategy, risk management, signal quality, market realism |
| `chart-analyst` | `.claude/agents/chart-analyst.md` | Candlestick patterns, chart formations, S/R levels, Fibonacci, pattern encoding |
| `market-patterns` | `.claude/agents/market-patterns.md` | SMC, order blocks, FVG, liquidity zones, session dynamics, inter-market correlations |
| `news-analyst` | `.claude/agents/news-analyst.md` | Macro events, sentiment pipeline (Claude API), news-to-signal design |
| `ml-specialist` | `.claude/agents/ml-specialist.md` | Generational training architecture, reward design, evolutionary logic, evaluation |

### Debate Protocol

1. **Spawn all 4 agents in parallel** — each analyzes the task from their domain
2. **Collect their positions** — what they recommend and why, and any blockers/concerns they raise
3. **Synthesize into a Decision Brief** — a short document with: agreed approach, tradeoffs acknowledged, open risks
4. **Present to the user** for approval before touching any code
5. **Only then implement**, using the `python-engineer` perspective as the implementation guide

### Example invocation pattern
When the user says "add RSI to the observation space":
- `forex-specialist`: Is RSI the right signal? What period? Should we use RSI(14) or RSI(2) for mean-reversion?
- `news-analyst`: Does RSI behavior change around news events? Should we mask RSI during high-impact events?
- `ml-specialist`: How does adding RSI affect observation dimensionality? Does it improve or hurt training?
- `python-engineer`: Where exactly in `env.py` does it go? What dtype? Does it need normalization?

The debate result is a concrete, agreed implementation plan — not a vague suggestion.
