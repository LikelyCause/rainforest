FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# The templates live outside /etc/nginx/templates so the image's own envsubst step doesn't render them with unset
# variables; 40-mlflow.sh renders them at start with an explicit variable list.
COPY nginx/default.conf.template nginx/mlflow.inc.template /etc/nginx/rainforest/
COPY --chmod=755 nginx/40-mlflow.sh /docker-entrypoint.d/40-mlflow.sh
ENV API_UPSTREAM=http://api:8000 \
    MLFLOW_UPSTREAM=http://mlflow:5000 \
    MLFLOW_ENABLED=1
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
