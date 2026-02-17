FROM python:3.11-slim-bookworm

# Install bash and jq
RUN apt-get update && apt-get install -y \
    bash \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Expose port
EXPOSE 8099

# Run
CMD ["/run.sh"]
