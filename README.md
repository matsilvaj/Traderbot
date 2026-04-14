# Traderbot RL (PPO + MetaTrader 5)

Projeto completo e modular para treinamento de um agente de trading com Reinforcement Learning (PPO) para BTC em `M1`, com foco em scalping.

## Aviso Importante

- O projeto vem configurado para **modo seguro** (`paper`) por padrão.
- Não execute em conta real sem validação adicional.
- `allow_live_trading` inicia como `false` em `config.yaml`.

## Arquitetura

- `src/traderbot/config.py`: Configuração centralizada (YAML + `.env`).
- `src/traderbot/data/mt5_loader.py`: Conexão e coleta de dados no MT5.
- `src/traderbot/features/engineering.py`: RSI, médias móveis, momentum, limpeza e normalização.
- `src/traderbot/env/trading_env.py`: Ambiente Gymnasium customizado com BUY/SELL/HOLD.
- `src/traderbot/rl/model_manager.py`: Treino, save/load de PPO (Stable-Baselines3).
- `src/traderbot/rl/backtest.py`: Avaliação em holdout + métricas.
- `src/traderbot/execution/mt5_executor.py`: Executor paper/live com trava de segurança.
- `src/traderbot/main.py`: Pipeline principal (train/backtest/run).

## Requisitos

- Python 3.10+
- Terminal MetaTrader 5 instalado
- Conta demo para testes de integração

Instalação:

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuração

1. Copie `.env.example` para `.env` e preencha credenciais (opcional):

```bash
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
MT5_PATH=
```

2. Ajuste `config.yaml`:

- `mt5.symbol`: ex. `BTCUSD`
- `mt5.timeframe`: `M1`
- `training.total_timesteps`
- `execution.mode`: `paper` ou `live`
- `execution.allow_live_trading`: `false` por segurança

## Como Rodar

Treinar + validar (backtest):

```bash
python -m traderbot.main --config config.yaml train
```

Treino robusto com múltiplas seeds:

```bash
python -m traderbot.main --config config.yaml train-multi
```

Rodar apenas backtest em modelo salvo:

```bash
python -m traderbot.main --config config.yaml backtest
```

Rodar loop de execução (paper/live):

```bash
python -m traderbot.main --config config.yaml run
```

Diagnóstico MT5 + restrições reais do broker:

```bash
python -m traderbot.main --config config.yaml check-mt5
```

## Métricas de Avaliação

O backtest salva:

- Lucro total
- Drawdown máximo
- Número de trades
- Número de entradas bloqueadas por risco/lote mínimo do broker
- Taxa de acerto

Arquivos gerados em `results/`:

- `*_metrics.json`
- `*_trades.csv`
- `*_equity.csv`

## Sobre Data Leakage

O pipeline evita vazamento de dados:

- split temporal (`train -> test`)
- normalização ajustada apenas no treino (`fit`) e aplicada no teste (`transform`)

## Logs

Logs em `logs/traderbot-rl.log`.

## Expansões Futuras

- stop-loss / take-profit dinâmicos
- gestão de risco por volatilidade
- múltiplos símbolos
- monitoramento em dashboard

## Simulação Realista

O projeto pode rodar um backtest mais próximo do broker real usando:

- `environment.simulation_initial_balance`
- `environment.use_broker_constraints`
- `environment.risk_per_trade`
- lote mínimo / step / contract size sincronizados via `check-mt5`

Com isso, treino e backtest podem refletir:

- saldo pequeno (ex.: `$50`)
- lote mínimo do broker
- bloqueio de trades quando o risco mínimo excede o alvo
