# Use Python 3.11 slim image
FROM python:3.11-slim AS python-base

# Poetry configuration
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    VENV_PATH="/app/.venv"

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Python configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY poetry.lock pyproject.toml ./

# Create empty README.md if it doesn't exist
RUN touch README.md

# Install dependencies
RUN poetry install --no-root --without dev

# Copy source code
COPY src/ src/
COPY run.py .

# Create data directory
RUN mkdir -p /app/data

# Environment variables (defaults, will be overridden by docker-compose)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=false \
    MODEL_TEMPERATURE=0.7 \
    MAX_TOKENS=2048 \
    DEFAULT_MODEL=mixtral-8x7b-32768

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "python", "run.py"]