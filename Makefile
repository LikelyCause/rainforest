.PHONY: up up-gpu gpu-check down logs ps train models test test-db psql sample-csv clean

up:            ## build and start everything (UI on :8080)
	docker compose up -d --build

up-gpu:        ## same, with the CUDA XGBoost build and the host's NVIDIA GPU (see README › GPU)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

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

psql:
	docker compose exec db psql -U $${POSTGRES_USER:-rainforest} $${POSTGRES_DB:-rainforest}

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
