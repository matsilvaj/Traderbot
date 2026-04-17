# Traderbot RL

Bot de trading com RL (PPO) para Hyperliquid, usando:
- timeframe `H1`
- ensemble por `majority vote`
- sizing dinamico por risco
- treino/backtest em CSV da Binance
- runtime com `paper_local`, `testnet` e `mainnet`

## Estado Atual

O fluxo principal validado hoje esta em:
- Hyperliquid
- `BTC`
- `1h`
- `decision_mode: ensemble_majority_vote`
- ensemble fixado com:
  - `ppo_btc_h1_s1`
  - `ppo_btc_h1_s21`
  - `ppo_btc_h1_s256`
  - `ppo_btc_h1_s512`
  - `ppo_btc_h1_s999`

## Requisitos

- Python `>= 3.10`
- Windows PowerShell
- carteira Hyperliquid/OKX Wallet ou outra wallet EVM compativel

Instalacao:

```powershell
pip install -r requirements.txt
```

## Launcher Desktop

Launcher moderno para operacao do bot sem precisar editar o `config.yaml`:

```powershell
python -m traderbot.launcher
```

No Windows, voce tambem pode abrir direto por:

```powershell
Trader Laucher.bat
```

O launcher permite:
- trocar entre `paper_local`, `testnet` e `mainnet`
- rodar `check`, `smoke` e `run` por botao
- acompanhar status, saldo, posicao e logs curtos
- usar kill switch para encerrar posicao no modo `exchange`

## Estrutura

- [config.yaml](C:\Users\Matheus\Documents\Traderbot\config.yaml): configuracao principal
- [COMANDOS_HYPERLIQUID.md](C:\Users\Matheus\Documents\Traderbot\COMANDOS_HYPERLIQUID.md): comandos prontos para testnet/mainnet
- [download_data.py](C:\Users\Matheus\Documents\Traderbot\download_data.py): baixa OHLCV da Binance
- [main.py](C:\Users\Matheus\Documents\Traderbot\src\traderbot\main.py): entrada principal do projeto
- [hl_loader.py](C:\Users\Matheus\Documents\Traderbot\src\traderbot\data\hl_loader.py): loader de dados da Hyperliquid
- [hl_executor.py](C:\Users\Matheus\Documents\Traderbot\src\traderbot\execution\hl_executor.py): executor paper/exchange
- [trading_env.py](C:\Users\Matheus\Documents\Traderbot\src\traderbot\env\trading_env.py): ambiente de treino e backtest

## Variaveis de Ambiente

Crie ou atualize o arquivo `.env` na raiz do projeto com:

```env
HL_WALLET_ADDRESS=0x...
HL_PRIVATE_KEY=0x...
```

Observacoes:
- `paper_local` nao precisa obrigatoriamente de envio real de ordem, mas o `check` e o `smoke` em `exchange` precisam dessas chaves
- nunca commite a private key

## Dados

O projeto hoje esta alinhado para CSV H1:

```powershell
python download_data.py --symbol BTC/USDT --market spot --timeframe 1h --years 5 --output data/binance_btcusdt_h1.csv
```

## Treino

Treino unico:

```powershell
python -m traderbot.main --config config.yaml train
```

Treino multi-seed:

```powershell
python -m traderbot.main --config config.yaml train-multi
```

## Backtest

Backtest:

```powershell
python -m traderbot.main --config config.yaml backtest
```

## Hyperliquid

Check da conexao:

```powershell
python -m traderbot.main --config config.yaml check-hyperliquid
```

Smoke test operacional:

```powershell
python -m traderbot.main --config config.yaml smoke-hyperliquid --network testnet --side buy --wait-seconds 3
```

Run do bot:

```powershell
python -m traderbot.main --config config.yaml run
```

Para nao precisar editar o `config.yaml`, use os comandos prontos em [COMANDOS_HYPERLIQUID.md](C:\Users\Matheus\Documents\Traderbot\COMANDOS_HYPERLIQUID.md).

## Modos Operacionais

- `paper_local`
  - usa saldo local do config
  - nao usa saldo da Hyperliquid para sizing
- `exchange + testnet`
  - usa saldo real da testnet
  - ideal para validacao operacional
- `exchange + mainnet`
  - usa saldo real da mainnet
  - recomendado somente depois de validar `check`, `smoke` e `run` na testnet

## Sizing e Execucao

O sizing operacional atual:
- usa saldo disponivel real do modo atual
- calcula `risk_amount = balance * risk_pct`
- calcula `target_notional = risk_amount / stop_loss_pct`
- respeita minimo de `10 USD`
- pula o trade se o minimo de `10 USD` violar o risco maximo

Separacao importante:
- `environment.slippage_pct`: slippage de modelagem/custo no bot
- `execution.order_slippage`: agressividade da ordem na exchange

## Observacoes

- o projeto nao usa mais MT5
- o fluxo ativo atual e H1 + Hyperliquid
- a alavancagem ainda nao esta implementada explicitamente no executor

## Comandos Rapidos

Testnet:

```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading check-hyperliquid
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side buy --wait-seconds 3
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading run
```

Mainnet:

```powershell
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading check-hyperliquid
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side buy --wait-seconds 3 --allow-mainnet
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading run
```
