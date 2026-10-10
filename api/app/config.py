from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql://rainforest:rainforest@localhost:5432/rainforest"
    readonly_role: str = "rainforest_ro"
    readonly_password: str = "rainforest_ro"
    data_dir: Path = Path("./data")

    job_workers: int = 2
    # Jobs running at once across every API worker process; each gets cpu_count // job_slots threads.
    job_slots: int = 2
    log_level: str = "info"
    db_pool_max: int = 10

    # Profiling / correlation sampling. Exact counts (nulls, uniques, min/max, mean)
    # always use every row; the heavier statistics below use a seeded random sample.
    profile_sample_rows: int = 500_000
    corr_sample_rows: int = 100_000
    corr_max_levels: int = 50
    copy_batch_rows: int = 100_000

    # Export memory budgets, as rows x output features x 8 bytes, for one transform chunk and for the fit
    # sample. Peak memory runs at a few times these (string columns, encoder working sets).
    export_chunk_mb: int = 256
    export_fit_mb: int = 512
    # Generated samples: rows x fields at most this (about 2 GB peak at the default).
    sample_max_cells: int = 60_000_000

    # SQL console guard rails
    sql_timeout_ms: int = 15_000
    sql_max_rows: int = 2_000
    sql_pool_max: int = 4

    # Sign-in. With no admin password the API serves only /health and refuses everything else.
    admin_user: str = Field("admin", validation_alias="RAINFOREST_ADMIN_USER")
    admin_password: str = Field("", validation_alias="RAINFOREST_ADMIN_PASSWORD")
    session_secret: str = Field("", validation_alias="RAINFOREST_SESSION_SECRET")
    allowed_hosts: str = Field("*", validation_alias="RAINFOREST_ALLOWED_HOSTS")

    # Empty: training runs are not logged to MLflow.
    mlflow_tracking_uri: str = ""
    git_sha: str = ""

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
