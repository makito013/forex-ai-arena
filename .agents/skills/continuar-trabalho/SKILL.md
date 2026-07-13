---
name: continuar-trabalho
description: Retoma o trabalho salvo por /pausar-trabalho no diretório atual (feito nesta ferramenta ou no Claude Code). Use quando o usuário disser /continuar-trabalho ou pedir pra continuar de onde parou.
---

# continuar-trabalho

Lê o checkpoint salvo por `/pausar-trabalho` para o diretório atual e retoma
o trabalho a partir dali.

## Passos

1. Rode:
   ```bash
   node ~/agentes-pipeline/claude/continuidade/core/state.js load \
     --cwd "$(pwd)"
   ```
2. Se a saída for exatamente `SEM_ESTADO_PAUSADO`: informe ao usuário que
   não há nenhuma pausa salva para este diretório, e pare por aqui.
3. Caso contrário, a saída é o resumo salvo, com um cabeçalho de metadados
   (`cwd`, `session_id`, `paused_at`) seguido do texto do resumo. Leia o
   conteúdo, mostre um resumo curto pro usuário do que estava em andamento
   (e há quanto tempo, com base em `paused_at`), e continue o trabalho a
   partir dos próximos passos descritos ali.
