from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator
from urllib.parse import quote_plus

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

_DATABASE_DISABLED = False
_DATABASE_DISABLED_REASON: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_timestamp(value: datetime | None) -> datetime:
    if value is None:
        return _utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        return normalized.replace("postgres://", "postgresql+psycopg://", 1)
    if normalized.startswith("postgresql://") and "+" not in normalized.split("://", 1)[0]:
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    return normalized


def _build_database_url_from_postgres_env() -> str | None:
    user = os.environ.get("POSTGRES_USER", "").strip()
    password = os.environ.get("POSTGRES_PASSWORD", "").strip()
    database = os.environ.get("POSTGRES_DB", "").strip()
    if not user or not password or not database:
        return None

    host = os.environ.get("POSTGRES_HOST", "").strip() or "localhost"
    port = os.environ.get("POSTGRES_PORT", "").strip() or "5432"
    return (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@"
        f"{host}:{port}/{quote_plus(database)}"
    )


def get_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        database_url = _build_database_url_from_postgres_env() or ""
    if not database_url:
        raise RuntimeError("DATABASE_URL nao definido no ambiente.")
    return _normalize_database_url(database_url)


def _compact_database_error(exc: Exception) -> str:
    message = str(exc).strip()
    background_marker = "\n(Background on this error"
    if background_marker in message:
        message = message.split(background_marker, 1)[0].strip()
    return f"{type(exc).__name__}: {message}"


def _disable_database(exc: Exception) -> None:
    global _DATABASE_DISABLED, _DATABASE_DISABLED_REASON
    if _DATABASE_DISABLED:
        return
    _DATABASE_DISABLED = True
    _DATABASE_DISABLED_REASON = _compact_database_error(exc)
    get_session_factory.cache_clear()
    get_engine.cache_clear()


def database_persistence_enabled() -> bool:
    return not _DATABASE_DISABLED


def database_unavailable_reason() -> str | None:
    return _DATABASE_DISABLED_REASON


class Base(DeclarativeBase):
    """Base declarativa do SQLAlchemy."""


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    total_profit: Mapped[float] = mapped_column(Float, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = get_database_url()
    engine_kwargs: dict[str, object] = {
        "pool_pre_ping": True,
    }
    if database_url.startswith("postgresql+psycopg://"):
        engine_kwargs["connect_args"] = {"connect_timeout": 5}
    return create_engine(database_url, **engine_kwargs)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables(engine: Engine | None = None) -> None:
    if not database_persistence_enabled():
        return
    database_engine = engine or get_engine()
    try:
        Base.metadata.create_all(database_engine)
    except Exception as exc:
        _disable_database(exc)
        raise


def insert_trade(
    *,
    symbol: str,
    side: str,
    price: float,
    amount: float,
    pnl: float | None = None,
    timestamp: datetime | None = None,
    session: Session | None = None,
) -> Trade | None:
    if not database_persistence_enabled():
        return None

    trade = Trade(
        timestamp=_coerce_timestamp(timestamp),
        symbol=str(symbol).strip().upper(),
        side=str(side).strip().lower(),
        price=float(price),
        amount=float(amount),
        pnl=None if pnl is None else float(pnl),
    )

    try:
        if session is not None:
            session.add(trade)
            session.flush()
            session.refresh(trade)
            return trade

        with session_scope() as managed_session:
            managed_session.add(trade)
            managed_session.flush()
            managed_session.refresh(trade)
            return trade
    except Exception as exc:
        _disable_database(exc)
        return None


def save_session_metrics(
    *,
    session_id: str,
    total_profit: float,
    win_rate: float,
    max_drawdown: float,
    timestamp: datetime | None = None,
    session: Session | None = None,
) -> Metric | None:
    if not database_persistence_enabled():
        return None

    metric = Metric(
        session_id=str(session_id).strip(),
        timestamp=_coerce_timestamp(timestamp),
        total_profit=float(total_profit),
        win_rate=float(win_rate),
        max_drawdown=float(max_drawdown),
    )

    try:
        if session is not None:
            session.add(metric)
            session.flush()
            session.refresh(metric)
            return metric

        with session_scope() as managed_session:
            managed_session.add(metric)
            managed_session.flush()
            managed_session.refresh(metric)
            return metric
    except Exception as exc:
        _disable_database(exc)
        return None


__all__ = [
    "Base",
    "Metric",
    "Trade",
    "create_tables",
    "database_persistence_enabled",
    "database_unavailable_reason",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "insert_trade",
    "save_session_metrics",
    "session_scope",
]
