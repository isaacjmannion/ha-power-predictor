FROM python:3.11-slim-bookworm

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
# Install packages one by one to see which one fails
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Flask==3.0.0 && \
    pip install --no-cache-dir requests==2.31.0 && \
    pip install --no-cache-dir gunicorn==21.2.0 && \
    pip install --no-cache-dir pytz==2023.3 && \
    pip install --no-cache-dir numpy==1.26.2 && \
    pip install --no-cache-dir pandas==2.1.4 && \
    pip install --no-cache-dir scikit-learn==1.4.0

# Copy application files
COPY app/ ./app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Expose port
EXPOSE 8099

# Run
CMD ["/run.sh"]
