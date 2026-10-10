.PHONY: up up-gpu up-keycloak gpu-check down logs ps train models mlflow-backfill test test-db psql sample-csv clean

# Tagged on MLflow runs (baked into the API image).
GIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null)
export GIT_SHA

up:            ## build and start everything (UI on :8080, MLflow at :8080/mlflow/)
	docker compose up -d --build

up-gpu:        ## same, with the CUDA XGBoost build and the host's NVIDIA GPU (see README › GPU)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# The Keycloak reference setup (README › Keycloak). Both URLs must be what browsers use to reach each service.
KEYCLOAK_URL ?= http://localhost:8081
RAINFOREST_URL ?= http://localhost:8080
up-keycloak:   ## start everything plus Keycloak, with OIDC sign-in (dev-only Keycloak; see README)
	KEYCLOAK_URL=$(KEYCLOAK_URL) RAINFOREST_URL=$(RAINFOREST_URL) \
	RAINFOREST_AUTH_MODE=oidc RAINFOREST_OIDC_PROVIDER_NAME=Keycloak \
	RAINFOREST_OIDC_ISSUER=$(KEYCLOAK_URL)/realms/rainforest \
	RAINFOREST_OIDC_DISCOVERY_URL=http://keycloak:8080/realms/rainforest/.well-known/openid-configuration \
	RAINFOREST_OIDC_CLIENT_ID=rainforest RAINFOREST_OIDC_CLIENT_SECRET=rainforest-dev-secret \
	RAINFOREST_OIDC_REDIRECT_URL=$(RAINFOREST_URL)/api/auth/oidc/callback \
	docker compose --profile keycloak up -d --build

gpu-check:     ## what the running API sees: GPU name, memory, driver, and whether XGBoost can use it
	docker compose exec api python -c "import json; from app.gpu import detect; print(json.dumps(detect(), indent=2))"

down:          ## stop (data volumes are kept)
	docker compose down

logs:
	docker compose logs -f api

ps:
	docker compose ps

MODEL ?= lightgbm
DEVICE ?= cpu
train:         ## make train EXPORT=1 [MODEL=lightgbm|xgboost|hist_gradient_boosting|random_forest|...] [DEVICE=cuda]
	docker compose exec api python -m app.modeling.train --export-id $(EXPORT) --model $(MODEL) --device $(DEVICE)

models:        ## list registered models and whether they're installed
	docker compose exec api python -m app.modeling.train --list-models

mlflow-backfill: ## log finished runs that have no MLflow run yet (trained before MLflow, or while it was down)
	docker compose exec api python -m app.modeling.mlflow_backfill

psql:          ## psql as the database owner; the db container's own POSTGRES_* values come from .env
	docker compose exec db sh -c 'psql -U "$$POSTGRES_USER" "$$POSTGRES_DB"'

sample-csv:    ## write a 200k-row sample CSV to ./sample.csv for testing uploads
	docker compose exec api python -m app.sample_data --rows 200000 --out /data/sample.csv
	docker compose cp api:/data/sample.csv ./sample.csv

test:          ## run API unit tests locally (needs `pip install -r api/requirements-dev.txt`); DB-backed tests skip
	cd api && pytest -q

TEST_PG_PORT ?= 55432
test-db:       ## run the full API suite against a throwaway Postgres container
	docker run -d --rm --name rainforest-testpg -e POSTGRES_PASSWORD=test -p 127.0.0.1:$(TEST_PG_PORT):5432 postgres:16 >/dev/null
	until docker exec rainforest-testpg pg_isready -q; do sleep 1; done
	cd api && RAINFOREST_TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:$(TEST_PG_PORT)/postgres pytest -q; \
	  status=$$?; docker rm -f rainforest-testpg >/dev/null; exit $$status

clean:         ## stop and DELETE all data volumes
	docker compose down -v
