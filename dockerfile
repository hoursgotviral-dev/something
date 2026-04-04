# ── Base Image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Dev A"
LABEL description="WhatsApp Sales RL – OpenEnv server"
LABEL version="1.0.0"

# ── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# ── Copy Project Files ───────────────────────────────────────────────────────
COPY . .

# ── Environment Settings ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose Port ──────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start Server (FINAL FIXED PORT) ──────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]