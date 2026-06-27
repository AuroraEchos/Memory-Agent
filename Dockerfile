FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app

COPY --chown=app:app pyproject.toml LICENSE README.md ./

RUN uv pip install --system \
    --extra-index-url https://pypi.org/simple \
    .

COPY --chown=app:app .chainlit/config.toml ./.chainlit/config.toml
COPY --chown=app:app memory_agent ./memory_agent
COPY --chown=app:app public ./public
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app chainlit_app.py embedding_server.py ./

RUN mkdir -p /app/models /app/.files /app/.chainlit \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000 8001

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
