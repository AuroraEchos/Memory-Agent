FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=1000
ARG APP_GID=1000
ARG TORCH_PACKAGE=torch==2.12.0+cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app

COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app memory_agent ./memory_agent
COPY --chown=app:app chainlit_app.py main.py ./

# sentence-transformers depends on torch; install the requested wheel first so
# CPU and GPU images do not accidentally resolve to a different runtime.
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_PACKAGE}" \
    && pip install -e . \
    && mkdir -p /app/data /app/models /app/.files /app/.chainlit \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
