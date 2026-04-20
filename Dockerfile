FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl && rm -rf /var/lib/apt/lists/*

ENV TZ=America/New_York

# Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY . .

# Non-root user for security
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

# Default: run continuously every 15 min
CMD ["python", "bot.py", "--loop", "--interval", "15"]
