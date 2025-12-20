
import pytest
from pydantic import SecretStr

from local_voice_agent.settings import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LVA_MODEL_NAME", "test-model")
    monkeypatch.setenv("LVA_API_KEY", "supersecret")

    s = Settings()
    assert s.model_name == "test-model"
    assert isinstance(s.api_key, SecretStr)
    assert s.api_key.get_secret_value() == "supersecret"


def test_settings_missing_required_raises(monkeypatch):
    # Ensure environment variables and .env are overridden by setting empty values
    monkeypatch.setenv("LVA_MODEL_NAME", "")
    monkeypatch.setenv("MODEL_NAME", "")
    monkeypatch.setenv("LVA_API_KEY", "")
    monkeypatch.setenv("API_KEY", "")

    with pytest.raises(ValueError):
        Settings()

