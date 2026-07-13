# Guia de Dados de Notícias para Forex AI Arena

Este documento explica como obter notícias históricas para treinar o modelo e notícias em tempo real para operação ao vivo, e o que é necessário para rodar cada opção.

---

## Parte 1 — Notícias Históricas (para treino)

### Opção A: GDELT Project ⭐ Recomendado (GRÁTIS)

O GDELT monitora notícias globais desde 1979 e é a maior base de dados de eventos de notícias aberta do mundo.

**O que oferece:**
- Cobertura desde 1979 (para Forex, o relevante é 2000+)
- Atualizado a cada 15 minutos
- Inclui: manchetes, fonte, país de origem, tonalidade (sentiment embutido)
- Cobertura em 65 idiomas

**Como obter:**
```bash
pip install gdeltdoc
```

```python
from gdeltdoc import GdeltDoc, Filters

gd = GdeltDoc()

# Buscar notícias sobre Fed/FOMC entre duas datas
filters = Filters(
    keyword="Federal Reserve interest rate",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

articles = gd.article_search(filters)
# articles é um DataFrame com: url, title, seendate, socialimage, domain, language, sourcecountry
```

**O que é necessário para rodar:**
- Python 3.8+
- `pip install gdeltdoc requests pandas`
- Conexão com internet (queries vão para a API do GDELT)
- Sem chave de API necessária

**Limitação:** Não tem full text dos artigos, apenas manchetes e metadados. Para sentiment, isso é suficiente — manda a manchete pro Claude.

---

### Opção B: Alpha Vantage News Sentiment (GRÁTIS com registro)

**O que oferece:**
- Notícias financeiras com sentiment score **já calculado** (não precisa chamar Claude para isso)
- Cobertura de pares forex, ações e commodities
- Histórico de até 10.000 artigos por query
- Score de relevância por ticker/par

**Como obter:**
1. Criar conta em: https://www.alphavantage.co/support/#api-key (grátis)
2. Instalar: `pip install alpha-vantage`

```python
import requests

API_KEY = "SUA_CHAVE_AQUI"

# Notícias sobre EURUSD com sentiment
url = (
    f"https://www.alphavantage.co/query"
    f"?function=NEWS_SENTIMENT"
    f"&tickers=FOREX:EUR"
    f"&time_from=20230101T0000"
    f"&time_to=20231231T2359"
    f"&limit=1000"
    f"&apikey={API_KEY}"
)

response = requests.get(url)
data = response.json()

# Cada item em data['feed'] tem:
# - title, summary, time_published
# - overall_sentiment_score: float [-1, 1]
# - overall_sentiment_label: Bearish/Neutral/Bullish
# - ticker_sentiment: score por par forex específico
```

**O que é necessário:**
- Chave API Alpha Vantage (grátis, limite: 25 calls/dia no plano free)
- `pip install requests pandas`
- Para volume maior: plano pago (~$50/mês)

---

### Opção C: Calendário Econômico Forex Factory (GRÁTIS, scraping)

O calendário econômico é a fonte mais importante para Forex — NFP, CPI, FOMC, etc.

**Dataset pronto (Kaggle):**
- Kaggle tem datasets históricos do Forex Factory já limpos e prontos
- Buscar: "forex factory economic calendar dataset" no Kaggle
- Contém: data, hora, moeda, evento, impacto (Low/Medium/High), Actual vs Forecast vs Previous

**Como usar o dataset:**
```python
import pandas as pd

# Após baixar o CSV do Kaggle:
calendar = pd.read_csv("forex_factory_calendar.csv", parse_dates=['datetime'])

# Filtrar eventos de alto impacto (Tier 1)
high_impact = calendar[calendar['impact'] == 'High']

# Criar score de surpresa: (actual - forecast) normalizado
high_impact['surprise'] = (high_impact['actual'] - high_impact['forecast']) / high_impact['previous'].abs()
```

**O que é necessário:**
- Conta no Kaggle (grátis)
- `pip install pandas`
- Sem conexão necessária após download

---

### Opção D: NewsAPI.org (GRÁTIS para desenvolvimento)

**O que oferece:**
- Artigos de Reuters, Bloomberg (limitado), FT, CNBC, MarketWatch
- Histórico de 1 mês no plano gratuito
- Histórico completo em planos pagos (~$449/mês para business)

**Como obter:**
1. Criar conta em: https://newsapi.org/ (grátis)
2. `pip install newsapi-python`

```python
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="SUA_CHAVE")

articles = newsapi.get_everything(
    q="Federal Reserve OR ECB OR interest rate OR inflation",
    from_param="2024-01-01",
    to="2024-06-01",
    language="en",
    sort_by="publishedAt"
)

for article in articles['articles']:
    print(article['publishedAt'], article['title'])
    # Mandar article['title'] + article['description'] pro Claude para sentiment
```

**O que é necessário:**
- Chave API NewsAPI (grátis: 100 calls/dia, 1 mês histórico)
- `pip install newsapi-python`
- Para histórico longo: plano pago

---

### Como Integrar Notícias Históricas com os CSVs de Preço

```
FLUXO DE PREPARAÇÃO DOS DADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Baixar notícias históricas (GDELT ou Alpha Vantage)
   → Salvar em: data/news/raw/YYYYMM_news.csv

2. Para cada manchete, chamar Claude Haiku para sentiment:
   → Salvar cache em: data/news/sentiment_cache.db (SQLite)
   → Schema: (hash_of_headline TEXT PRIMARY KEY, score REAL, tier INT, timestamp DATETIME)

3. Criar DataFrame de sentiment por intervalo de tempo:
   → Agrupar scores por hora (ou pelo intervalo do seu CSV de preço)
   → Weighted average: eventos mais recentes têm peso maior

4. Fazer merge com o CSV de preço pelo timestamp:
   → df_price.merge(df_sentiment, left_index=True, right_index=True, how='left')
   → Preencher NaN com 0.0 (neutro quando sem notícia)

5. O resultado é o DataFrame final para treino: preço + indicadores + sentiment
```

---

## Parte 2 — Notícias em Tempo Real (para operação ao vivo)

### Opção A: Alpha Vantage News Real-Time ⭐ Mais Simples (GRÁTIS)

A mesma API de histórico funciona em real-time — basta não passar filtro de data.

```python
import requests, time

def fetch_latest_sentiment(ticker="FOREX:EUR", api_key="SUA_CHAVE"):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}&limit=10&apikey={api_key}"
    )
    data = requests.get(url).json()
    articles = data.get('feed', [])
    if not articles:
        return 0.0
    
    # Score das últimas 10 notícias (decaimento por tempo)
    now = pd.Timestamp.now(tz='UTC')
    weighted_score = 0.0
    total_weight = 0.0
    
    for article in articles:
        pub_time = pd.Timestamp(article['time_published']).tz_convert('UTC')
        age_hours = (now - pub_time).total_seconds() / 3600
        weight = max(0.0, 1.0 - age_hours / 24)  # Decai a zero em 24h
        score = float(article.get('overall_sentiment_score', 0.0))
        weighted_score += score * weight
        total_weight += weight
    
    return weighted_score / total_weight if total_weight > 0 else 0.0

# Polling loop (a cada 5 minutos)
while True:
    score = fetch_latest_sentiment()
    print(f"EUR Sentiment: {score:.3f}")
    time.sleep(300)
```

**O que é necessário:**
- Chave API Alpha Vantage (grátis)
- Limite: 25 calls/dia (grátis) → polling a cada ~60min máximo
- Para polling frequente: plano pago ($50/mês = 75 calls/min)

---

### Opção B: RSS Feeds de Notícias (GRÁTIS, sem limite)

RSS feeds são públicos e sem rate limit — ótimos para monitoramento contínuo.

```bash
pip install feedparser schedule
```

```python
import feedparser
import time
from anthropic import Anthropic

RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_forex":    "https://feeds.reuters.com/news/wealth",
    "ft_economics":     "https://www.ft.com/rss/home/uk",
    "marketwatch":      "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
}

claude = Anthropic()

def analyze_headline_with_claude(headline: str) -> float:
    msg = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                "Financial analyst. Read this Forex headline and return ONLY a float "
                "from -1.0 (strongly bearish) to 1.0 (strongly bullish). No text, just the number.\n\n"
                f"Headline: {headline}"
            )
        }]
    )
    try:
        return max(-1.0, min(1.0, float(msg.content[0].text.strip())))
    except:
        return 0.0

def poll_rss_feeds():
    recent_scores = []
    for name, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:  # 3 mais recentes de cada fonte
            score = analyze_headline_with_claude(entry.title)
            recent_scores.append(score)
    
    return sum(recent_scores) / len(recent_scores) if recent_scores else 0.0

# Loop de 5 em 5 minutos
while True:
    sentiment = poll_rss_feeds()
    print(f"Aggregated Sentiment: {sentiment:.3f}")
    time.sleep(300)
```

**O que é necessário:**
- `pip install feedparser`
- Chave API Anthropic (para Claude Haiku)
- Custo estimado: ~$0.002 por ciclo de 5 min (12 headlines × Claude Haiku) = ~$8.64/mês rodando 24/7
- Sem chave de API de terceiros necessária

---

### Opção C: Calendário Econômico em Tempo Real

Para saber quando vem o próximo NFP, CPI ou FOMC antes de abrir posição:

```bash
pip install investpy  # ou a alternativa: tradingeconomics
```

**Alternativa gratuita: scraping do Forex Factory:**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_upcoming_events(days_ahead=7):
    """Retorna eventos econômicos dos próximos N dias."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    response = requests.get(url)
    events = response.json()
    
    df = pd.DataFrame(events)
    # Filtrar apenas impacto alto (vermelho no Forex Factory)
    high_impact = df[df['impact'] == 'High']
    return high_impact[['title', 'country', 'date', 'forecast', 'previous']]

# Uso:
events = get_upcoming_events()
print(events)
#  title          country  date                 forecast  previous
#  Non-Farm...    USD      2024-02-02T13:30:00  185K      216K
```

**O que é necessário:**
- `pip install requests beautifulsoup4 pandas`
- Sem chave de API

---

## Parte 3 — O que é necessário para rodar em tempo real (checklist completo)

### Hardware Mínimo
| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | Qualquer dual-core | 4+ cores (para rodar múltiplos agentes) |
| RAM | 8 GB | 16 GB (múltiplos agentes em paralelo) |
| Internet | 10 Mbps estável | Qualquer conexão estável |
| GPU | Não necessário | Opcional (acelera treino, não inferência) |

### Chaves de API Necessárias
| Serviço | Uso | Custo | Onde obter |
|---------|-----|-------|------------|
| **Anthropic (Claude)** | Análise de sentiment | ~$5–15/mês para uso moderado | console.anthropic.com |
| **Alpha Vantage** | Notícias com sentiment pronto | Grátis (25 calls/dia) | alphavantage.co |
| **yfinance** | Dados de preço em tempo real | Grátis | Já integrado no projeto |

### Variáveis de Ambiente Necessárias
```bash
# Adicionar ao ~/.zshrc ou .env no projeto:
export ANTHROPIC_API_KEY="sk-ant-..."
export ALPHA_VANTAGE_API_KEY="SEU_KEY"
```

### Pacotes Python a Instalar
```bash
pip install anthropic feedparser gdeltdoc newsapi-python schedule
```

### Fluxo de Operação em Tempo Real
```
LOOP DE OPERAÇÃO AO VIVO:
━━━━━━━━━━━━━━━━━━━━━━━

A cada 1 minuto (ou no timeframe do agente):
  1. yfinance → busca candle atual (OHLCV)
  2. Alpha Vantage / RSS → busca notícias recentes
  3. Claude Haiku → converte headlines em sentiment score
  4. ForexEnv._next_observation() → monta vetor de observação
  5. PPO model.predict(obs) → decide ação (Hold/Buy/Sell/Close)
  6. ForexEnv.step(action) → simula execução (paper trading)
  7. Streamlit UI → atualiza gráfico + posição + P&L em tempo real
```

---

## Parte 4 — Estratégia Recomendada para Este Projeto

**Para treino inicial (já disponível):**
- Use os CSVs históricos de preço em `data/historical/` — você já tem tudo
- Baixe o calendário econômico histórico do Kaggle (dataset Forex Factory)
- Use Alpha Vantage para complementar com sentiment histórico

**Para o passo seguinte (quando o agente estiver treinado):**
1. RSS feeds gratuitos para notícias em tempo real
2. Alpha Vantage para sentiment score automático (sem chamar Claude a cada notícia)
3. Reserve Claude para análises mais ricas quando houver evento Tier 1

**Custo total estimado para operação real:**
- Anthropic API: ~$5–10/mês (uso moderado com Haiku)
- Alpha Vantage: $0 (plano grátis suficiente para começo)
- NewsAPI: $0 (RSS feeds substituem bem)
- **Total: ~$5–10/mês**
