from __future__ import annotations

import os
import sys


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "nao"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw if raw in choices else default


def main() -> int | None:
    config_path = os.getenv("TRADERBOT_CONFIG", "config.yaml")
    network = _env_choice("TRADERBOT_NETWORK", "mainnet", {"mainnet", "testnet"})
    execution_mode = _env_choice("TRADERBOT_EXECUTION_MODE", "exchange", {"exchange", "paper_local"})
    command = str(os.getenv("TRADERBOT_COMMAND", "run") or "run").strip()

    args = [
        "traderbot.main",
        "--config",
        config_path,
        "--network-override",
        network,
        "--execution-mode-override",
        execution_mode,
    ]
    if _env_bool("TRADERBOT_ALLOW_LIVE_TRADING", True):
        args.append("--allow-live-trading")
    args.append(command)

    sys.argv = args
    from traderbot.main import main as traderbot_main

    return traderbot_main()


if __name__ == "__main__":
    raise SystemExit(main())
