# One image for the API, the dashboard and the simulation scripts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RECOMMENDER_HOME=/app \
    WEB_CONCURRENCY=4

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 recommender \
    && mkdir -p /app/data /app/results \
    && chown -R recommender /app/data /app/results
USER recommender

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# uvicorn reads the number of worker processes from WEB_CONCURRENCY.
CMD ["uvicorn", "recommender.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
