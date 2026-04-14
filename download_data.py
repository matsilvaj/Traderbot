from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import ccxt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "A biblioteca 'ccxt' não está instalada. Adicione-a ao ambiente antes de rodar este script."
    ) from exc


TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa histórico M1 da Binance em CSV.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Par a ser baixado. Ex: BTC/USDT")
    parser.add_argument(
        "--market",
        choices=["spot", "linear"],
        default="spot",
        help="Spot da Binance ou futuros perpétuos lineares (USDT-M).",
    )
    parser.add_argument("--timeframe", default="1m", help="Timeframe do OHLCV. Ex: 1m")
    parser.add_argument(
        "--years",
        type=float,
        default=3.0,
        help="Quantidade aproximada de anos para trás a partir de agora.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Data inicial ISO opcional. Ex: 2023-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/binance_btcusdt_m1.csv",
        help="Arquivo CSV de saída.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limite por requisição. 1000 é um valor seguro para Binance.",
    )
    return parser.parse_args()


def build_exchange(market: str):
    if market == "linear":
        return ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def resolve_since_ms(args: argparse.Namespace) -> int:
    if args.since:
        dt = pd.Timestamp(args.since)
    else:
        now = pd.Timestamp.now(tz="UTC")
        dt = now - pd.Timedelta(days=int(args.years * 365))
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return int(dt.timestamp() * 1000)


def fetch_all_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int, limit: int) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Timeframe não suportado neste script: {timeframe}")

    step_ms = TIMEFRAME_MS[timeframe]
    now_ms = exchange.milliseconds()
    rows: list[list[float]] = []
    seen = set()
    next_since = since_ms

    while next_since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit)
        if not batch:
            break

        added = 0
        for candle in batch:
            ts = int(candle[0])
            if ts not in seen:
                rows.append(candle)
                seen.add(ts)
                added += 1

        last_ts = int(batch[-1][0])
        next_since = last_ts + step_ms

        print(
            f"Baixados {len(rows)} candles até {datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat()}",
            flush=True,
        )

        if added == 0:
            break

        sleep_ms = max(int(getattr(exchange, "rateLimit", 0)), 50)
        time.sleep((sleep_ms + 50) / 1000.0)

    if not rows:
        raise RuntimeError("Nenhum candle retornado pela exchange.")

    df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    since_ms = resolve_since_ms(args)
    exchange = build_exchange(args.market)

    try:
        markets = exchange.load_markets()
        if args.symbol not in markets:
            raise ValueError(f"Símbolo '{args.symbol}' não encontrado na Binance ({args.market}).")

        df = fetch_all_ohlcv(
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            limit=args.limit,
        )
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            close_fn()

    df.to_csv(output_path, index=False)
    print(f"CSV salvo em {output_path.resolve()} com {len(df)} linhas.", flush=True)


if __name__ == "__main__":
    main()
