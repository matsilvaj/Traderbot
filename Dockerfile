FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.10-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    TRADERBOT_CONFIG=config.yaml \
    TRADERBOT_NETWORK=mainnet \
    TRADERBOT_EXECUTION_MODE=exchange \
    TRADERBOT_ALLOW_LIVE_TRADING=true

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md requirements.txt requirements-ui.txt config.yaml ./
COPY src ./src
COPY data ./data
COPY models ./models

RUN mkdir -p /app/data /app/models /app/logs /app/results

CMD ["python", "-u", "-m", "traderbot.headless_entry"]
