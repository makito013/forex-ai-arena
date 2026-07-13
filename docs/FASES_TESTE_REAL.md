# Forex AI Arena — Roadmap até o Teste Real

Documento das fases para levar um agente do treino até operar em condições reais
(dados ao vivo, custos reais, plataforma oficial) **sem arriscar dinheiro de verdade
antes da hora**.

> **Regra de ouro:** cada fase só começa quando a anterior fecha os critérios de avanço.
> Pular fase = testar contra uma realidade que não existe e perder dinheiro depois.

---

## Estado atual da plataforma

O que já está construído e funcionando:

- **Treino com Liberdade Total** (modos *Deep Evolutionary* 🧬 e *Goal-Oriented* 🎯):
  cada agente escolhe par e timeframe aleatórios de `data/historical/` a cada epoch,
  em janelas aleatórias (`Max Candles per Epoch`) para equilibrar a duração.
- **Gestão de risco real** no `ForexEnv` (e visível ao agente como observação):
  checagem de margem, proibição de all-in (`max_margin_usage_pct`), stop-out do broker
  (`stop_out_level`), proteção de saldo negativo e falência (equity < 5%).
- **Drawdown na recompensa**: o agente é punido por aprofundar drawdown e por ultrapassar
  o limite configurado (`max_drawdown_limit`); a evolução promove quem tem **fitness
  ajustado ao risco**, não lucro puro.
- **Leverage por sessão** (UI), continuação a partir de **Base Agents**, e **💾 Stop & Save**.
- **Arena com "Match Training Conditions"**: leverage, janela de candles e modo
  determinístico — para testar o agente nas MESMAS condições do treino.
- **✅ Custos reais (Fase 0)**: spread + comissão + swap (ver abaixo).

### Os 3 descasamentos que zeram um agente bom (já mitigados)
1. **Leverage diferente** entre treino e Arena → resolvido com o campo de leverage na Arena.
2. **Janela curta no treino × dataset inteiro na Arena** → resolvido com `Max Candles` na Arena.
3. **Custo fictício** → resolvido na Fase 0.

---

## Fase 0 — Modelo de custo realista ✅ CONCLUÍDA

**Objetivo:** trocar a taxa fictícia de 5% sobre a margem por custos de broker real,
para que qualquer teste seguinte signifique alguma coisa.

**O que foi feito** (`config.yaml` → seção `costs`, `FinancialEngine`, `ForexEnv`):
- **Spread** pago no round-trip (compra no *ask*, vende no *bid*), por par, com pip correto
  por ativo (0.0001 forex / 0.01 JPY e ouro). Pares USD-base convertem o custo para USD.
- **Comissão** por lote (round-turn).
- **Swap** por noite que a posição fica aberta (trade intradiário não paga).
- Flag `costs.enabled` liga/desliga (false volta ao modelo antigo só para comparação).

**Defaults (conta Standard):** EURUSD 1.0 pip, ouro 20 pts, USDJPY 1.2 pips, comissão 0,
swap -$0.50/lote/noite. Para **ECN/Raw**: spread ~0.1 pip + `commission_per_lot: 7.0`.

**Validação:** EURUSD 0.01 lote ≈ $0.10/trade, ouro ≈ $0.20, USDJPY ≈ $0.08 — batem com
a realidade. Contabilidade do trade: `saldo = inicial − custo_abertura + pnl − swap`.

**Critérios de avanço (faça antes da Fase 1):**
- [ ] Definir o tipo de conta-alvo (Standard ou ECN) e ajustar `config.yaml`.
- [ ] Re-validar os agentes salvos na **Arena com custo real ligado**. Os que dependiam
      da taxa irreal vão desabar — descarte-os.
- [ ] **Re-treinar do zero** os sobreviventes (eles aprenderam contra o custo errado),
      com janelas grandes (50k+) e drawdown alvo ≤ 25%.
- [ ] Ter ≥1 agente que lucra **com custo real, em par/período que NÃO viu no treino**
      (teste de overfit na Arena).

---

## Fase 1 — Paper trade em Python (sanity check)

**Objetivo:** rodar o agente sobre candles ao vivo (yfinance), montando a observação
de 29 features, simulando preenchimento com spread realista e logando os trades.
Rápido e grátis — mas limitado, porque **yfinance ≠ feed do broker**.

**O que precisa ser construído:**
- Loop agendado que puxa o último candle do par, recalcula os indicadores (TA-Lib) e
  monta a mesma observação do `ForexEnv`.
- `model.predict(obs)` → simula a ordem com spread/comissão do modelo da Fase 0.
- Log de trades + curva de equity (CSV/painel).

**Critérios de avanço:**
- [ ] O agente roda dias sem erro de shape/observação.
- [ ] O comportamento ao vivo bate qualitativamente com o backtest (não vira
      "compra tudo" nem "nunca opera").
- [ ] Curva de equity coerente com o esperado do treino.

> **Limite honesto:** isto NÃO é teste real — é só para pegar bugs de integração antes
> de conectar no MT5. Não tire conclusões de lucro daqui.

---

## Fase 2 — Conta DEMO no MT5 (o teste real) 🎯

**Objetivo:** rodar o agente em conta **demo de um broker real** via o pacote Python
`MetaTrader5`. Dá dados ao vivo do feed do broker, **spread/swap/comissão reais**,
execução real (slippage, latência) — tudo com dinheiro fake.

**Arquitetura:** o cérebro continua em Python (modelo PPO). O pacote `MetaTrader5`
(`pip install MetaTrader5`) conecta no terminal MT5 rodando:
- lê preço ao vivo do broker e reconstrói a observação idêntica à do treino;
- lê o estado real da conta (`AccountInfoDouble`: saldo, equity, margem);
- envia ordens reais para a conta demo (`order_send`).

**O que precisa ser construído:**
- `src/live/mt5_bridge.py`: conexão, leitura de rates, reconstrução da observação,
  envio/gestão de ordens, sincronização de timeframe.
- Mapeamento de símbolos (ex.: `XAUUSD` do broker ↔ contract size/pip do projeto).
- Reconciliação: o saldo/posições vêm da conta MT5, não do `ForexEnv` simulado.
- Logging de cada decisão e fill para auditoria.

**Critérios de avanço:**
- [ ] Observação reconstruída ao vivo == observação do treino (mesmos indicadores, mesma escala).
- [ ] Ordens executam na demo e batem com o que o agente decidiu.
- [ ] Custos reais do broker ≈ os parametrizados na Fase 0 (ajustar `config.yaml` ao real).

---

## Fase 3 — Avaliação prolongada → dinheiro real

**Objetivo:** rodar a demo por **semanas** e comparar o lucro mensal real contra o do treino,
antes de cogitar capital real.

**Critérios para considerar dinheiro real (todos):**
- [ ] Lucro mensal positivo e consistente na demo por **várias semanas** (não um pico de sorte).
- [ ] **Max drawdown** dentro do limite que você definiu (ex.: ≤ 25%) em condições reais.
- [ ] Resultado da demo coerente com o backtest/Arena (sem surpresa grande = sem overfit oculto).
- [ ] Plano de risco definido: tamanho de lote, perda máxima diária, kill-switch.

> Começar com o **mínimo** de capital e o **menor lote** possível, mesmo após a demo passar.

---

## Decisões pendentes

| Decisão | Por que importa | Status |
|---|---|---|
| Tipo de conta: **Standard** vs **ECN/Raw** | Define spread/comissão no `config.yaml` e que tipo de agente vale treinar | ⏳ definir |
| Corretora-alvo | Define os custos reais e se o calendário MT5 está disponível | ⏳ definir |
| Pares-foco para operar | Concentrar treino/validação nos pares que vai operar de verdade | ⏳ definir |

---

## A verdade dura (não esqueça)

- **Drawdown de 60% não é resultado bom** — é frágil; sobrevive em janela curta de sorte e
  quebra no longo prazo. Mire ≤ 25%.
- A **Arena é seu detector de overfit**: teste sempre em par/período que o agente NÃO treinou.
  Se só lucra no que viu, é decoreba, não estratégia.
- **yfinance ≠ broker.** Conclusão de lucro só vale a partir da Fase 2 (demo real).
- O custo real é implacável com quem **opera demais** — o melhor agente provavelmente
  opera pouco e bem, não o que mais movimenta.

---

## Referência rápida — chaves de config

```yaml
trading:
  leverage: 500
  max_margin_usage_pct: 0.5     # sem all-in
  stop_out_level: 0.5           # margin call do broker
  max_drawdown_limit: 0.30      # punição de risco na recompensa
costs:
  enabled: true                 # custos reais (false = taxa fictícia de 5%)
  default_spread_pips: 1.0
  commission_per_lot: 0.0       # ECN: ~7.0
  swap_per_lot_per_night: -0.5
  spreads_pips:
    XAUUSD: 20.0
    USDJPY: 1.2
```
