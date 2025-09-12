# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package configuration for dependency installation
COPY pyproject.toml README.md ./

# Install build dependencies and package
RUN pip install --upgrade pip setuptools wheel build
RUN pip install -e .[production,api]

# Production stage
FROM python:3.11-slim as production

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    ARCHITECTURE_MODE=unified

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/docs /app/logs /app/chroma_db && \
    chown -R appuser:appuser /app

# Copy source code with proper ownership
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser docs /app/docs
COPY --chown=appuser:appuser data /app/data

# Switch to non-root user
USER appuser

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Environment variable documentation
# Core Configuration:
# - GOOGLE_API_KEY: Google Gemini API key (required)
# - OPENAI_API_KEY: OpenAI API key (optional)
# - GROQ_API_KEY: Groq API key (optional)
# 
# Architecture Settings:
# - ARCHITECTURE_MODE: unified (default), cli, api
# - ENGINE_MAX_CONCURRENT_REQUESTS: 10 (default)
# - ENGINE_REQUEST_TIMEOUT: 120 (default)
# 
# Performance Settings:
# - ENABLE_RESPONSE_CACHE: true (default)
# - CACHE_TTL_SECONDS: 3600 (default)
# - ASYNC_PROCESSING_ENABLED: true (default)
#
# Example usage:
# docker run -d -p 8000:8000 \
#   -e GOOGLE_API_KEY=your_key_here \
#   -e ARCHITECTURE_MODE=unified \
#   support-deflect-bot:latest

# Start the application using the new unified architecture
CMD ["uvicorn", "src.support_deflect_bot.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
