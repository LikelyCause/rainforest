.PHONY: up down logs ps train models test psql sample-csv clean

up:            ## build and start everything (UI on :8080)
	docker compose up -d --build

down:          ## stop (data volumes are kept)
	docker compose down

logs:
	docker compose logs -f api

ps:
	docker compose ps

MODEL ?= lightgbm
train:         ## make train EXPORT=1 [MODEL=lightgbm|xgboost|hist_gradient_boosting|random_forest|...]
	docker compose exec api python -m app.modeling.train --export-id $(EXPORT) --model $(MODEL)

models:        ## list registered models and whether they're installed
	docker compose exec api python -m app.modeling.train --list-models

psql:
	docker compose exec db psql -U $${POSTGRES_USER:-rainforest} $${POSTGRES_DB:-rainforest}

sample-csv:    ## write a 200k-row sample CSV to ./sample.csv for testing uploads
	docker compose exec api python -m app.sample_data --rows 200000 --out /data/sample.csv
	docker compose cp api:/data/sample.csv ./sample.csv

test:          ## run API unit tests locally (needs `pip install -r api/requirements-dev.txt`)
	cd api && pytest -q

clean:         ## stop and DELETE all data volumes
	docker compose down -v
