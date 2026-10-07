from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql://rainforest:rainforest@localhost:5432/rainforest"
    readonly_role: str = "rainforest_ro"
    readonly_password: str = "rainforest_ro"
    data_dir: Path = Path("./data")
    default_targets: str = "column_a,column_b"

    job_workers: int = 2
    log_level: str = "info"
    db_pool_max: int = 10

    # Profiling / correlation sampling. Exact counts (nulls, uniques, min/max, mean)
    # always use every row; the heavier statistics below use a seeded random sample.
    profile_sample_rows: int = 500_000
    corr_sample_rows: int = 100_000
    corr_max_levels: int = 50
    copy_batch_rows: int = 100_000

    # SQL console guard rails
    sql_timeout_ms: int = 15_000
    sql_max_rows: int = 2_000

    @property
    def target_list(self) -> list[str]:
        return [t.strip() for t in self.default_targets.split(",") if t.strip()]

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
