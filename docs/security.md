# Security

What Rainforest protects, how, and what you need to set up yourself. The deployment steps are in
[Deployment](deployment.md#team-deployment-checklist).

## Choose the right sign-in mode

Use `RAINFOREST_AUTH_MODE=oidc` for anything but a single person or a trusted network. Pick mode lets anyone who
reaches the UI act as any user, including a lead.

## Network

| Service | Listens on |
|---|---|
| `ui` (`UI_PORT`, 8080) | every interface, so you can reach it over the LAN |
| `api` (`API_PORT`) and `db` (`DB_PORT`) | localhost only |
| `mlflow` | no published port |
| `keycloak` profile (`KEYCLOAK_PORT`, 8081) | every interface |

Every API route needs the session cookie, file downloads included, except `/health`, sign-in and profile pictures.
The MLflow UI at `/mlflow/` needs it too.

On plain HTTP, the session cookie crosses the network unencrypted. Anywhere beyond a trusted LAN, put the UI behind a
TLS proxy that sets `X-Forwarded-Proto`.

Set `RAINFOREST_ALLOWED_HOSTS` to the names and addresses you use, so a DNS-rebound name is refused outright.

## Requests that change data

- Browser requests that change data must be same-origin: `Origin` or `Referer` must match the host.
  `http://localhost:5173` and `http://127.0.0.1:5173` are also accepted, for `npm run dev`.
- A request body must be `application/json`, or `multipart/form-data` for uploads. This blocks cross-site form posts.

## Sessions

- The session cookie holds only the user id and their session version. It is signed (HMAC) with the session secret.
- It is `HttpOnly` and `SameSite=Strict`, and lasts 7 days. It is `Secure` when the request came over HTTPS.
- Every request checks that the user is still active and the version current. So deactivation and role changes apply
  at once.
- Rotating `RAINFOREST_SESSION_SECRET` signs everyone out. When it is empty, the secret is generated on first start
  and kept in `/data/session_secret`.

## OIDC sign-in

- Every sign-in uses PKCE (S256), `state` and `nonce`.
- The state travels in a short-lived signed cookie. It is `SameSite=Lax`, which the provider's redirect back needs,
  and scoped to `/api/auth/oidc`.
- The PKCE verifier and nonce are derived from the state with the server's secret, so nothing else is stored.
- ID tokens are checked against the provider's published keys (asymmetric algorithms only), issuer, audience, expiry
  and nonce.
- Provider tokens are never stored or sent to the browser.
- Users are matched by the provider's subject id, never by email.

The [reference Keycloak](deployment.md#reference-keycloak-development-only) is for development only. Its client
secret and passwords are public.

## Deployment keys

- A key opens only its own deployment's predict call. It isn't a session and reads nothing else.
- Only its SHA-256 and first 8 characters are stored, and it is compared in constant time. It is shown once and
  never logged. The SQL console's role can't read the keys table at all.
- The predict call with a key is exempt from the same-origin check: a browser never sends a key on its own, so it
  can't be forged cross-site. With a session it is checked like any other write.
- Revoking a key stops it at once. See [Keys](deployments.md#keys).

## Profile pictures

- Pictures are open at a URL naming the picture's hash, because the sign-in page shows them.
- Uploads are decoded and re-encoded as WebP, never stored as sent.
- Pictures are served with `nosniff` and a sandboxing content policy.

## Privacy between users

- An assessment is private to its assessor while the version is assessing. See
  [Assessing](team-workflow.md#assessing).
- A personal experiment is private while its version is assessing. See
  [Privacy](experiments-and-training.md#privacy).
- The SQL console can't read private work. See [SQL](sql.md#privacy).
- MLflow has no access control, so runs in private experiments wait. See
  [MLflow](mlflow.md#private-experiments-wait).
- Viewers are read-only. The API refuses their writes with `403`, MLflow included.
