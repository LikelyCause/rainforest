# Reference identity provider for the `keycloak` Compose profile. The realm is baked in (not bind-mounted), so it
# also works against a remote Docker daemon.
FROM quay.io/keycloak/keycloak:26.7.5
COPY realm-rainforest.json /opt/keycloak/data/import/realm-rainforest.json
