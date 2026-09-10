# Fluxo de trabalho do time

Convenção de versionamento adotada pelo grupo na Fase 3, dando continuidade ao fluxo usado na
Fase 2.

## Branches

| Branch | Propósito |
|---|---|
| `main` | Código estável e revisado. Nunca recebe commit direto. |
| `feat/<assunto>` | Nova funcionalidade ou etapa do pipeline (ex.: `feat/pipeline-sklearn`). |
| `fix/<assunto>` | Correção pontual (ex.: `fix/leakage-id-escola`). |
| `docs/<assunto>` | Documentação, README, apresentação. |

## Ciclo

1. Abrir a branch a partir de `main` atualizada.
2. Commits pequenos e descritivos, em português, no imperativo
   (`adiciona validação cruzada agrupada por UF`).
3. Abrir Pull Request descrevendo **o que muda**, **por que muda** e **como validar**.
4. Revisão por pelo menos um integrante antes do merge.
5. Merge com `--no-ff`, preservando o contexto da branch no histórico.

## Documentação das decisões analíticas

Toda decisão de modelagem que altere o resultado é registrada em três lugares:

- no **código**, como comentário no ponto onde a decisão é aplicada;
- no **notebook** correspondente, com a evidência que a sustenta;
- no **README**, na seção de etapas de modelagem.

Decisões já documentadas desta forma: remoção do histórico por escola (reanonimização anual do
`ID_ESCOLA`), escolha da safra de 2025 como base do modelo municipal (ancoragem das metas na linha
de base de 2023), restrição do universo a alunos presentes e bloqueio das três camadas de vazamento.
