# Comandos Hyperliquid

Todos os comandos abaixo funcionam sem editar o `config.yaml`, usando overrides via CLI.

## Testnet

### Check
```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading check-hyperliquid
```

### Smoke Test Buy
```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side buy --wait-seconds 3
```

### Smoke Test Sell
```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side sell --wait-seconds 3
```

### Run do Bot
```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading run
```

## Mainnet

### Check
```powershell
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading check-hyperliquid
```

### Smoke Test Buy
```powershell
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side buy --wait-seconds 3 --allow-mainnet
```

### Smoke Test Sell
```powershell
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading smoke-hyperliquid --side sell --wait-seconds 3 --allow-mainnet
```

### Run do Bot
```powershell
python -m traderbot.main --config config.yaml --network-override mainnet --execution-mode-override exchange --allow-live-trading run
```

## Override opcional do slippage operacional da ordem

Use só se quiser testar outro valor para a agressividade da ordem na exchange.

### Exemplo na testnet
```powershell
python -m traderbot.main --config config.yaml --network-override testnet --execution-mode-override exchange --allow-live-trading --order-slippage-override 0.05 run
```

## VPS / Docker

Subir apenas o painel Telegram:

```powershell
docker compose up -d traderbot
```

No painel Telegram, use:

```text
/sinais 24
```

Esse comando mostra se os ultimos ciclos ficaram em HOLD, geraram BUY/SELL, foram bloqueados por filtro/risco, ou abriram entrada.

Subir o runtime headless direto, com os mesmos defaults do launcher/painel:

```powershell
docker compose --profile runtime up -d traderbot-runtime
```

Para trocar rede ou modo na VPS:

```env
TRADERBOT_NETWORK=mainnet
TRADERBOT_EXECUTION_MODE=exchange
TRADERBOT_ALLOW_LIVE_TRADING=true
```
