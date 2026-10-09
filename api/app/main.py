from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import auth, db, jobs
from .config import get_settings
from .routers import datasets, explore, exports, fields, query, runs
from .semantics import option_catalog

logging.basicConfig(
    level=get_settings().log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    for d in (s.uploads_dir, s.datasets_dir, s.exports_dir, s.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    auth.session_secret()
    if not auth.admin_password():
        log.warning(auth.NO_PASSWORD)
    db.bootstrap()
    db.mark_interrupted_jobs()
    yield
    jobs.shutdown()
    db.close_pool()


app = FastAPI(title="Rainforest API", version="1.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(auth.SecurityMiddleware)
# The UI is served same-origin through nginx; CORS only matters for `npm run dev` on :5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(fields.router)
app.include_router(query.router)
app.include_router(exports.router)
app.include_router(runs.router)
app.include_router(explore.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    db.fetch_one("SELECT 1 AS ok")
    return {"status": "ok"}


@app.get("/api/options", tags=["meta"])
def options() -> dict:
    return option_catalog()
