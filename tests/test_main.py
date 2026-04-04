import importlib

import dotenv

import app.main


def test_main_loads_dotenv_when_env_file_exists(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    def _fake_load_dotenv(path):
        calls["count"] += 1

    monkeypatch.setattr(dotenv, "load_dotenv", _fake_load_dotenv)

    importlib.reload(app.main)

    assert calls["count"] == 1


def test_main_ignores_env_loading_errors(monkeypatch):
    def _raise_exists(self):
        raise RuntimeError("boom")

    monkeypatch.setattr("pathlib.Path.exists", _raise_exists)

    # Should not raise because app.main handles env-loading errors.
    importlib.reload(app.main)
