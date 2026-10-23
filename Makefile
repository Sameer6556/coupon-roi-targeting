# Every target is a thin wrapper around the `couponroi` CLI, which also works on Windows
# without make (e.g. `couponroi all`).

.PHONY: install data build eda causal churn uplift policy drift export all synthetic test lint app api docker mlflow notebooks clean

install:
	pip install -e ".[dev]"

data:
	couponroi data
build:
	couponroi build
eda:
	couponroi eda
causal:
	couponroi causal
churn:
	couponroi churn
uplift:
	couponroi uplift
policy:
	couponroi policy
drift:
	couponroi drift
export:
	couponroi export

all:
	couponroi all

synthetic:   ## whole pipeline on generated data (what CI runs)
	couponroi all --synthetic --no-mlflow

test:
	pytest

lint:
	ruff check src tests app api
	ruff format --check src tests app api

app:
	streamlit run app/Home.py

api:
	uvicorn api.main:app --reload

docker:
	docker build -t coupon-roi-api .
	docker run --rm -p 8000:8000 coupon-roi-api

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

notebooks:
	python notebooks/build_notebooks.py

eval-ask:    ## needs an LLM endpoint, see src/couponroi/ask.py
	couponroi ask-eval

clean:
	rm -rf data/warehouse.duckdb mlruns mlflow.db .pytest_cache .ruff_cache
