FROM python:3.12.3-slim

# EasyPanel (and similar panels) often default to building `./Dockerfile` at repo root.
# This Dockerfile builds the FastAPI agent service container.

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
RUN uv sync --no-install-project --no-dev

# Plotly (para gráficos en chat) y Kaleido (PNG opcional)
RUN pip install --no-cache-dir plotly kaleido

COPY src/agents/ ./agents/
COPY src/core/ ./core/
COPY src/memory/ ./memory/
COPY src/schema/ ./schema/
COPY src/service/ ./service/
COPY src/voice/ ./voice/
COPY media/ ./media/
COPY web/ ./web/
COPY src/run_service.py .

EXPOSE 8080
CMD ["python", "run_service.py"]
