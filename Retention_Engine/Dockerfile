#  Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a local prefix so we can copy just them later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


#  Runtime stage 
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/       ./app/
COPY data/      ./data/
COPY frontend/  ./frontend/

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# Health check — Vercel equivalent of /health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
