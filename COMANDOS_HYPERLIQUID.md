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
