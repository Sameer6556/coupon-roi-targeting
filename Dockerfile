# Scoring API image. Build from the repo root after `couponroi all` has produced artifacts/.
#   docker build -t coupon-roi-api .
#   docker run -p 8000:8000 coupon-roi-api      ->  http://localhost:8000/docs
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY artifacts/models/ artifacts/models/
COPY artifacts/app/app.duckdb artifacts/app/app.duckdb

ENV COUPONROI_ROOT=/srv PYTHONUNBUFFERED=1
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
