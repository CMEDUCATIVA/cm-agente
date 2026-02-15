FROM python:3.12.3-slim

# EasyPanel (and similar panels) often default to building `./Dockerfile` at repo root.
# This Dockerfile builds the FastAPI agent service container.

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY src/agents/ ./agents/
COPY src/core/ ./core/
COPY src/memory/ ./memory/
COPY src/schema/ ./schema/
COPY src/service/ ./service/
COPY src/voice/ ./voice/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Plotly (para gráficos en chat) y Kaleido (PNG opcional)
RUN pip install --no-cache-dir plotly kaleido

COPY media/ ./media/
COPY web/ ./web/
COPY src/run_service.py .

EXPOSE 8080
CMD ["python", "run_service.py"]
