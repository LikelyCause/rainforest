.PHONY: up up-gpu up-keycloak gpu-check down logs ps train models mlflow-backfill test test-db psql sample-csv clean

# Tagged on MLflow runs (baked into the API image).
GIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null)
export GIT_SHA

# .env is the configuration: Compose reads it on every command, COMPOSE_FILE and COMPOSE_PROFILES included.
up:            ## build and start everything as .env describes (UI on :8080, MLflow at :8080/mlflow/)
	docker compose up -d --build

# A setting as Compose sees it: the environment (or `make VAR=...`) first, then .env.
setting = $(or $($(1)),$(shell sed -n 's/^[[:space:]]*$(1)=//p' .env 2>/dev/null | tail -n 1 | sed -e 's/^"\(.*\)"$$/\1/' -e "s/^'\(.*\)'$$/\1/"))
comma := ,

gpu_files = $(or $(call setting,COMPOSE_FILE),docker-compose.yml)
up-gpu:        ## `make up` plus the CUDA XGBoost image, for this run only (keep it: COMPOSE_FILE in .env; README › GPU)
	COMPOSE_FILE=$(if $(findstring docker-compose.gpu.yml,$(gpu_files)),$(gpu_files),$(gpu_files):docker-compose.gpu.yml) \
	docker compose up -d --build
	@$(if $(findstring docker-compose.gpu.yml,$(call setting,COMPOSE_FILE)),,echo "The GPU image applied to this run only; to keep it for \`make up\`, add COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml to .env.")

# The reference Keycloak's sign-in settings, for each one .env leaves empty; none when .env names its own issuer.
# Both URLs must be what browsers use to reach each service.
kc_url = $(or $(call setting,KEYCLOAK_URL),http://localhost:8081)
rf_url = $(or $(call setting,RAINFOREST_URL),http://localhost:8080)
KEYCLOAK_OIDC = RAINFOREST_AUTH_MODE=oidc RAINFOREST_OIDC_PROVIDER_NAME=Keycloak \
	RAINFOREST_OIDC_ISSUER=$(kc_url)/realms/rainforest \
	RAINFOREST_OIDC_DISCOVERY_URL=http://keycloak:8080/realms/rainforest/.well-known/openid-configuration \
	RAINFOREST_OIDC_CLIENT_ID=rainforest RAINFOREST_OIDC_CLIENT_SECRET=rainforest-dev-secret \
	RAINFOREST_OIDC_REDIRECT_URL=$(rf_url)/api/auth/oidc/callback
kc_fill = $(if $(call setting,RAINFOREST_OIDC_ISSUER),,$(foreach d,$(KEYCLOAK_OIDC),$(if $(call setting,$(firstword $(subst =, ,$(d)))),,$(d))))
kc_profiles = $(call setting,COMPOSE_PROFILES)
kc_on = $(filter keycloak,$(subst $(comma), ,$(kc_profiles)))
kc_profile = COMPOSE_PROFILES=$(if $(kc_on),$(kc_profiles),$(if $(kc_profiles),$(kc_profiles)$(comma))keycloak)
kc_keep = $(strip $(if $(kc_on),,$(kc_profile)) $(kc_fill))
up-keycloak:   ## `make up` plus the dev-only Keycloak, filling the sign-in settings .env leaves empty (README › Keycloak)
	@case "$(call setting,RAINFOREST_AUTH_MODE)" in ""|oidc) ;; \
	  *) echo "RAINFOREST_AUTH_MODE is $(call setting,RAINFOREST_AUTH_MODE); set it to oidc, or remove it, in .env." >&2; exit 1;; esac
	env $(kc_fill) $(kc_profile) docker compose up -d --build
	@$(if $(kc_keep),printf '%s\n' "These applied to this run only. To keep them for \`make up\` add them to .env:" $(foreach d,$(kc_keep),"  $(d)"))

gpu-check:     ## what the running API sees: GPU name, memory, driver, and whether XGBoost can use it
	docker compose exec api python -c "import json; from app.gpu import detect; print(json.dumps(detect(), indent=2))"

down:          ## stop and remove the containers, the Keycloak profile's included (data volumes are kept)
	docker compose --profile '*' down

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

mlflow-backfill: ## log every finished run that has no MLflow run (e.g. one trained while MLflow was down or turned off)
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
	@docker rm -f rainforest-testpg >/dev/null 2>&1 || true
	docker run -d --rm --name rainforest-testpg -e POSTGRES_PASSWORD=test -p 127.0.0.1:$(TEST_PG_PORT):5432 postgres:16 >/dev/null
	trap 'docker rm -f rainforest-testpg >/dev/null' EXIT; \
	  for i in $$(seq 60); do docker exec rainforest-testpg pg_isready -q && break; sleep 1; done; \
	  cd api && RAINFOREST_TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:$(TEST_PG_PORT)/postgres pytest -q

clean:         ## stop and DELETE all data volumes
	docker compose --profile '*' down -v
