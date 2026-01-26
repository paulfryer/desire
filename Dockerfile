# DESIRE Research - Reproducibility Container
# Python 3.11 with all dependencies for full reproduction

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Install optional dependencies for visualization
RUN pip install streamlit panel plotly umap-learn || true

# Copy the entire project
COPY . .

# Create output directories
RUN mkdir -p outputs/agents outputs/test_results outputs/figures outputs/extended_analysis

# Set default seed for reproducibility
ENV DESIRE_SEED=42

# Default command: run full reproduction
CMD ["bash", "scripts/reproduce_all.sh"]

# Alternative entry points:
# Train agents only:
#   docker run desire-research python experiments/train_agents.py
#
# Run tests only:
#   docker run desire-research python experiments/run_tests.py
#
# Generate figures:
#   docker run desire-research python scripts/generate_figures.py
#
# Interactive shell:
#   docker run -it desire-research bash
