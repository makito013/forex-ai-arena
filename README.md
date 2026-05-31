# 📈 Forex AI Arena

Uma plataforma de simulação gamificada para treinamento de agentes de Inteligência Artificial (Reinforcement Learning) no mercado Forex. As IAs competem operando com dinheiro fictício (Paper Trading) em um ambiente com regras reais de corretagem e alavancagem.

---

## 🚀 Como Iniciar (Passo a Passo)

### 1. Pré-requisitos
Antes de começar, você precisa ter instalado na sua máquina:
- **Python 3.10** ou superior.
- **Ollama** (Opcional, mas recomendado): Para usar a análise de sentimento baseada em notícias e calendário econômico rodando localmente (Llama 3).

### 2. Instalação e Setup
Abra o terminal, navegue até a pasta do projeto e siga os comandos abaixo para criar o ambiente virtual e instalar as dependências:

```bash
# Clone o repositório (se ainda não o fez)
# git clone https://github.com/makito013/forex-ai-arena.git
cd forex-ai-arena

# Crie o ambiente virtual Python
python3 -m venv venv

# Ative o ambiente virtual
# No Mac/Linux:
source venv/bin/activate
# No Windows:
# venv\Scripts\activate

# Instale as bibliotecas necessárias
pip install -r requirements.txt
```

### 3. Configurações e Variáveis de Ambiente (Envs)
Este projeto foi desenhado para rodar **sem a necessidade de chaves de API pagas** inicialmente:
- **Dados do Mercado**: Usamos a biblioteca `yfinance` que extrai dados gratuitamente do Yahoo Finance. Não exige API Key.
- **Inteligência Artificial (LLM)**: O projeto usa o `Ollama` rodando no `localhost:11434`. Sem custos ou chaves da OpenAI.

**Onde configurar as regras do jogo?**
Toda a configuração de alavancagem, corretagem e ativos fica no arquivo `config.yaml`:
```yaml
trading:
  leverage: 100            # Alavancagem padrão (1:100)
  brokerage_fee_pct: 0.05  # Corretagem de 5% sobre a margem investida
  standard_lot_size: 100000
  default_micro_lots: 0.01 # Lotes de 1.000 unidades
  base_currency: "USD"

assets:
  pairs:
    - symbol: "EURUSD=X"   # Você pode adicionar novos pares do yfinance aqui
      name: "EUR/USD"

arena:
  initial_balance: 10000.0 # Saldo inicial fictício para cada nova IA
```
*Sinta-se à vontade para editar este arquivo antes de treinar os agentes.*

---

## 🕹️ Como Usar a Plataforma

A plataforma possui duas peças centrais: o **Dashboard Visua**l e o **Motor de Treinamento**.

### A. Rodando o Dashboard (Streamlit)
Para ver os gráficos, testar a matemática financeira e acompanhar o Ranking (Leaderboard) das IAs, rode o seguinte comando no terminal (com o `venv` ativado):

```bash
streamlit run app.py
```
*Isso abrirá automaticamente uma aba no seu navegador acessando `http://localhost:8501`.*

### B. Treinando novas IAs (Agora direto pelo Portal!)
Para treinar novos agentes com gráficos passados e testar diferentes estratégias e ativos (como Ouro ou USD/CHF), você não precisa mais rodar scripts de terminal!

1. Abra o painel do Streamlit (`http://localhost:8501`).
2. Vá até a aba **"Training Arena"**.
3. Selecione o **Par de Moeda** desejado (ex: Ouro, EUR/USD).
4. Selecione o **Período Histórico** (ex: 1 Mês, 1 Ano). Gráficos de 1 Ano irão acelerar o aprendizado, pois testam o robô contra ciclos completos de mercado.
5. Selecione a quantidade de Agentes.
6. Clique em **Start Training Session** e acompanhe a barra de progresso!

*(Nota: O script antigo `train_agent.py` ainda existe na pasta como um exemplo de linha de comando, mas a UI é a forma oficial de operar).*

---

## 📂 Estrutura do Projeto
- `app.py`: Interface de usuário.
- `train_agent.py`: Script principal de treinamento.
- `config.yaml`: Central de configurações do ambiente e taxas.
- `database.db`: Banco local SQLite (gerado automaticamente) contendo histórico de trades e agentes.
- `src/engine/env.py`: As regras e o ambiente Gymnasium onde a IA vive.
- `src/engine/financial.py`: As matemáticas de lucro, margem e taxa de corretagem.
- `src/db_models.py`: Modelos das tabelas do banco de dados.

Qualquer dúvida técnica ou contexto extra, a pasta possui os arquivos `GEMINI.md` e `CLAUDE.md` com explicações da arquitetura para serem inseridas em IAs auxiliares.
