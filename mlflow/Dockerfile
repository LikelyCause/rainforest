# The official image has no Postgres driver; its SQLAlchemy 2.1 uses psycopg 3 for postgresql:// URIs.
FROM ghcr.io/mlflow/mlflow:v3.17.0
RUN pip install --no-cache-dir "psycopg[binary]==3.2.4"
COPY start.py /usr/local/bin/mlflow-start
ENTRYPOINT ["python", "/usr/local/bin/mlflow-start"]
