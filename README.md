# Plano de Implementação: Plataforma de Treinamento de IAs para Forex

## Objetivo
Criar uma plataforma em Python para simulação, competição e treinamento de IAs (Agentes) no mercado Forex. O objetivo final é treinar um modelo autônomo perfeito para trading. O sistema envolverá múltiplas IAs competindo com dinheiro fictício (Paper Trading), consumindo dados gráficos em tempo real, notícias e calendário econômico, e lidando com cálculos realistas de corretagem e alavancagem.

## Arquitetura e Tecnologias
- **Linguagem**: Python 3.10+
- **Interface Gráfica (UI)**: Streamlit (Dashboard interativo para visualização dos gráficos, ranking das IAs e histórico de operações).
- **Banco de Dados**: SQLite (Leve, local e eficiente para armazenar o histórico de ordens, posições abertas e saldo de cada IA).
- **Fontes de Dados**:
  - *Preços*: `yfinance` para dados em tempo real.
  - *Notícias/Calendário*: Integração com APIs gratuitas (ex: Finnhub ou NewsAPI) ou Web Scraping (Investing.com).
- **Motor de Inteligência Artificial (Abordagem Híbrida - Recomendada)**:
  - *Processamento de Texto (Notícias)*: LLMs rodando localmente via **Ollama** (ex: Llama 3) para ler notícias e o calendário econômico e extrair um "Score de Sentimento" (Bullish/Bearish).
  - *Tomada de Decisão (Trading)*: **Reinforcement Learning (RL)** usando bibliotecas como `Stable Baselines3` ou algoritmos genéticos (`NEAT`). Múltiplos agentes de RL receberão os dados do gráfico + o Score de Sentimento do LLM e tomarão ações (Buy, Sell, Hold). Aqueles com maior lucro sobrevivem e "reproduzem/aprendem".

## Lógica de Negociação (Market Simulator)
As IAs não operam no mercado real, mas em um ambiente simulado com as seguintes regras de corretagem e alavancagem:
- **Tamanho do Lote**: 1 Lote Padrão = 100.000 unidades da moeda base. O sistema operará com micro lotes (0.01 = 1.000 unidades).
- **Alavancagem**: Padrão Forex (ex: 1:100 ou 1:500 configurável). Isso define a Margem Necessária para abrir a posição.
- **Corretagem (Fee)**: 5% sobre a margem investida (valor utilizado para abrir a posição).
  - *Exemplo*: Para abrir 0.01 lotes (1.000 unidades) de EUR/USD a 1.1000 com alavancagem 1:100:
    - Valor Total da Posição = $1,100.
    - Margem Investida = $1,100 / 100 = $11.00.
    - Corretagem (5%) = $11.00 * 0.05 = $0.55.
  - A operação já começa negativa no valor da corretagem + spread.

## Estrutura do Projeto
```text
forex-ai-arena/
├── config.yaml          # Configuração (pares, alavancagem, corretagem)
├── requirements.txt
├── database.db          # Arquivo SQLite gerado automaticamente
├── src/
│   ├── data/            # Fetchers (yfinance, News, Calendário)
│   ├── engine/          # Motor de simulação de trading (calcula lucros, corretagem)
│   ├── ai/              # Agentes de RL e integração com LLM local (Ollama)
│   └── db_models.py     # Schemas do SQLite (Agentes, Posições, Histórico)
└── app.py               # Dashboard Streamlit
```

## Funcionalidades do Dashboard (Portal)
1. **Leaderboard**: Ranking das IAs pelo saldo/ROI (Retorno sobre Investimento).
2. **Gráficos em Tempo Real**: Visualização dos pares com as entradas e saídas (Buy/Sell) feitas pela IA selecionada.
3. **Painel de Posições**:
   - Posições Abertas (IA, Par, Tipo, Tamanho, Preço de Abertura, Tempo, Margem, Lucro Atual).
   - Histórico de Fechamento (Resultados, Preço de Fechamento, Tempo de Fechamento).
4. **Terminal de Contexto**: Visualização das últimas notícias e dados do calendário econômico que as IAs estão consumindo.

## Passos de Implementação
1. **Setup de Dados e Motor Financeiro**: Implementar a lógica matemática de cálculo de Lotes, Pip, Alavancagem e Corretagem de 5%.
2. **Banco de Dados**: Configurar o SQLite com tabelas para `Agents`, `OpenPositions` e `TradeHistory`.
3. **Pipeline de Dados AI**: Integrar busca de notícias e configurar a chamada para LLM local (Ollama) para gerar "Sentiment Scores".
4. **Ambiente de RL (Gym)**: Criar o ambiente virtual de trading onde os Agentes viverão e aprenderão, processando velas (candles) e sentimentos.
5. **Dashboard Streamlit**: Desenvolver a interface para observação e controle da competição de IAs.
