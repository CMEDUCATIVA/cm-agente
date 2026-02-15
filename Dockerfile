FROM python:3.11-slim

# EasyPanel (and similar panels) often default to building `./Dockerfile` at repo root.
# This Dockerfile builds the FastAPI agent service container.

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
# Ensure editable install can find the package at /app/src
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Plotly (para gráficos en chat) y Kaleido (PNG opcional)
RUN pip install --no-cache-dir plotly kaleido

COPY media/ ./media/
COPY web/ ./web/
COPY src/run_service.py .

EXPOSE 8080
CMD ["python", "run_service.py"]
