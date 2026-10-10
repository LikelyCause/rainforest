"""docker-compose.yml passes every API setting through from .env, with the same defaults as config.py."""
import re
from pathlib import Path

import pytest
import yaml

from app.config import Settings

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
NOT_PASSED = {"readonly_role", "git_sha"}  # fixed by design; baked into the image at build time
# Compose computes these differently from the API's local-development defaults.
OWN_DEFAULTS = {"database_url", "data_dir", "readonly_password", "admin_password", "session_secret",
                "mlflow_tracking_uri"}


@pytest.fixture(scope="module")
def api_env() -> dict[str, str]:
    if not COMPOSE.exists():
        pytest.skip("docker-compose.yml is not next to the api directory")
    return yaml.safe_load(COMPOSE.read_text())["services"]["api"]["environment"]


def _env_name(name: str, field) -> str:
    return field.validation_alias if isinstance(field.validation_alias, str) else name.upper()


def test_every_setting_reaches_the_api_container(api_env):
    missing = [_env_name(n, f) for n, f in Settings.model_fields.items()
               if n not in NOT_PASSED and _env_name(n, f) not in api_env]
    assert missing == []


def test_compose_defaults_match_config(api_env):
    for name, field in Settings.model_fields.items():
        if name in NOT_PASSED | OWN_DEFAULTS:
            continue
        m = re.fullmatch(r"\$\{(\w+):-(.*)\}", str(api_env[_env_name(name, field)]))
        if m and m.group(2) and isinstance(field.default, int):
            assert int(m.group(2)) == field.default, name
