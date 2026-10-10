# Reference identity provider for `make up-keycloak`. The realm is baked in (not bind-mounted) so the profile also
# works against a remote Docker daemon.
FROM quay.io/keycloak/keycloak:26.7.5
COPY realm-rainforest.json /opt/keycloak/data/import/realm-rainforest.json
