# Headless Engine Architecture

## What changed now

- The engine now writes a shared runtime state file independently of the runtime guard.
- The launcher polls that state file in addition to parsing stdout.
- This gives the GUI a first-class read path for a headless engine running outside PySide6.

## Recommended split

### 1. Engine service

Keep all trading responsibilities in a standalone process:

- market data fetch
- model inference
- order execution
- runtime heartbeat/state snapshots
- risk controls and kill switch enforcement

Suggested command:

```bash
python -m traderbot.main --config config.yaml run
```

This process should be able to run under:

- Windows service
- Docker container
- systemd/supervisor
- launcher-managed local process

### 2. Shared read model

Treat logs as human diagnostics only. The GUI should read a structured state snapshot instead:

- file-based JSON for local mode
- local socket or HTTP endpoint for remote mode

Suggested snapshot contract:

```json
{
  "status": "running",
  "network": "mainnet",
  "symbol": "BTC",
  "last_heartbeat_at": "2026-04-18T12:00:00+00:00",
  "last_cycle_payload": {
    "event_code": "execution.position_held",
    "bar_timestamp": "2026-04-18T11:00:00+00:00",
    "position_label": "LONG",
    "available_to_trade": 250.0,
    "final_action": 0.82
  }
}
```

### 3. Control channel

For a true remote client, introduce a small command channel separated from stdout:

- `GET /state`
- `POST /actions/check`
- `POST /actions/smoke`
- `POST /actions/close-position`
- `POST /actions/stop-runtime`

Local-only alternative:

- TCP socket on `127.0.0.1`
- newline-delimited JSON requests/responses

### 4. GUI responsibilities

The launcher should become a thin client:

- render state
- trigger commands
- show notifications
- read structured health/errors

The GUI should not own:

- execution timing
- retry policy
- order safety logic
- protection state

## Migration plan

1. Keep the current CLI engine as the single source of truth.
2. Move GUI reads from log parsing to `runtime_state_*.json` wherever possible.
3. Add a small command API/socket in a dedicated module such as `traderbot.engine_api`.
4. Make the launcher call that API instead of spawning subprocesses directly.
5. Keep stdout logs only for operator inspection and post-mortem analysis.
