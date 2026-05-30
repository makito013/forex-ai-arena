# Forex AI Arena - Contexto e Instruções para o Gemini

## Visão Geral do Projeto
O **Forex AI Arena** é uma plataforma de gamificação e simulação de trading focada no treinamento de agentes de Inteligência Artificial usando Aprendizado por Reforço (Reinforcement Learning - RL). O objetivo é colocar múltiplas IAs competindo no mercado Forex (Paper Trading) para descobrir a melhor estratégia sob condições rigorosas de margem e corretagem.

## Arquitetura e Tecnologias
- **Linguagem**: Python 3.10+
- **Bibliotecas Principais**: `yfinance` (dados do mercado), `pandas`, `stable-baselines3` (PPO RL agent), `gymnasium` (ambiente customizado), `SQLAlchemy` (ORM), `streamlit` (UI/Dashboard).
- **Banco de Dados**: SQLite local (`database.db`). Armazena os Agentes, Posições Abertas e Histórico de Trades.
- **Integração LLM**: Abordagem híbrida. Um LLM local (via Ollama) lê notícias/calendário econômico para gerar um "Sentiment Score", que é então passado como observação para o agente RL.

## Motor Financeiro (Regras Críticas)
Qualquer modificação ou cálculo de lucro/prejuízo (PnL) deve seguir estritamente o `FinancialEngine` (`src/engine/financial.py`):
1. **Lote Padrão**: 100.000 unidades. As IAs operam em micro lotes por padrão (0.01 = 1.000 unidades).
2. **Alavancagem**: Definida em `config.yaml` (padrão 1:100). Usada para calcular a *Margem Investida*.
3. **Corretagem (Fee)**: 5% sobre a *Margem Investida*. Essa taxa é deduzida IMEDIATAMENTE ao abrir a posição, fazendo com que o agente já inicie a operação no negativo. O PnL real é o lucro bruto menos essa taxa.

## Estrutura de Diretórios
- `src/engine/env.py`: Ambiente `ForexEnv` (compatível com Gymnasium). É aqui que o agente RL interage com os preços e toma ações (0: Hold, 1: Buy, 2: Sell, 3: Close).
- `src/db_models.py`: Schemas do SQLAlchemy.
- `app.py`: Dashboard em Streamlit.
- `train_agent.py`: Pipeline principal de treinamento que inicializa o ambiente, treina o agente PPO e salva os resultados.
- `config.yaml`: Arquivo de configuração (pares, taxas, alavancagem).

## Comandos Úteis
- **Treinar nova IA**: `python train_agent.py`
- **Rodar UI**: `streamlit run app.py`

## Diretrizes de Manutenção (Para o Gemini)
- Ao debugar erros de dimensionamento no RL (ex: `ValueError` no Numpy), verifique o casting explícito de tipos provenientes do Pandas DataFrame dentro da função `_next_observation` do ambiente.
- Não altere a lógica de punição imediata do `brokerage_fee` no `step()` do `ForexEnv` a menos que solicitado. Isso é parte essencial da dificuldade do treinamento do agente.
- Mantenha a interface do Streamlit focada em observação e leitura de dados do SQLite. A UI não deve executar rotinas de treino pesadas de forma síncrona.
