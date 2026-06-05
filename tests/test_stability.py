"""
Comprehensive stability test suite for ByteCLI.

Covers unit tests for all core managers, state machine, i18n, PID management,
model switcher, recording FSM, and corner cases.  All heavy dependencies
(Whisper, PyTorch, GTK, D-Bus) are mocked so these tests run on any machine
with Python 3.10+ and pytest.

Run:
    /usr/bin/python3 -m pytest tests/test_stability.py -v
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
import types
import uuid
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable without GTK / D-Bus / Whisper installed.
# We mock the gi module at the top level for tests that import GTK-dependent
# code indirectly.
# ---------------------------------------------------------------------------

# ===================================================================
# 1. ConfigManager tests
# ===================================================================

from bytecli.service.config_manager import ConfigManager
from bytecli.constants import DEFAULT_CONFIG, WHISPER_MODELS


class TestConfigManager:
    """Unit tests for ConfigManager load / save / validate cycle."""

    def _make_manager(self, tmp_path: str) -> ConfigManager:
        config_file = os.path.join(tmp_path, "config.json")
        return ConfigManager(config_file=config_file)

    def test_load_creates_default_when_missing(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        config = mgr.load()
        assert config["model"] == DEFAULT_CONFIG["model"]
        assert config["device"] == DEFAULT_CONFIG["device"]
        assert os.path.isfile(mgr.config_file)

    def test_sensevoice_hidden_without_model_dir(self, monkeypatch):
        import importlib
        import bytecli.constants as constants

        monkeypatch.delenv("BYTECLI_SENSEVOICE_ONNX_MODEL_DIR", raising=False)
        reloaded = importlib.reload(constants)
        try:
            assert "zh_fast" not in reloaded.VISIBLE_INFERENCE_PROFILES
        finally:
            importlib.reload(constants)

    def test_remote_profile_set_limits_visible_models(self, monkeypatch, tmp_path):
        import importlib
        import bytecli.constants as constants

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: object()
            if name in {"funasr", "torchaudio", "qwen_asr"}
            else real_find_spec(name),
        )
        monkeypatch.setenv("BYTECLI_PROFILE_SET", "remote")
        monkeypatch.setenv("BYTECLI_DATA_DIR", str(tmp_path))
        reloaded = importlib.reload(constants)
        try:
            assert reloaded.DEFAULT_CONFIG["model"] == "experimental_qwen"
            assert reloaded.VISIBLE_INFERENCE_PROFILES == (
                "experimental_qwen",
                "fun_asr_nano",
            )
            assert tuple(reloaded.WHISPER_MODELS) == (
                "fun_asr_nano",
                "sherpa_sensevoice",
                "sherpa_funasr_nano",
                "experimental_qwen",
            )
        finally:
            monkeypatch.delenv("BYTECLI_PROFILE_SET", raising=False)
            monkeypatch.delenv("BYTECLI_DATA_DIR", raising=False)
            importlib.reload(constants)

    def test_save_and_reload(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        custom = copy.deepcopy(DEFAULT_CONFIG)
        custom["model"] = "tiny"
        custom["language"] = "zh"
        mgr.save(custom)

        mgr2 = self._make_manager(str(tmp_path))
        reloaded = mgr2.load()
        assert reloaded["model"] == "tiny"
        assert reloaded["language"] == "zh"

    def test_corrupt_file_recovery(self, tmp_path):
        config_file = os.path.join(str(tmp_path), "config.json")
        with open(config_file, "w") as f:
            f.write("NOT VALID JSON {{{")

        mgr = ConfigManager(config_file=config_file)
        config = mgr.load()
        # Should fall back to defaults.
        assert config["model"] == DEFAULT_CONFIG["model"]
        # Backup should exist.
        assert os.path.isfile(config_file + ".bak")

    def test_validate_valid_config(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        errors = mgr.validate(copy.deepcopy(DEFAULT_CONFIG))
        assert errors == []

    def test_validate_invalid_model(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["model"] = "nonexistent"
        errors = mgr.validate(bad)
        assert "model" in errors

    def test_validate_invalid_device(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["device"] = "tpu"
        errors = mgr.validate(bad)
        assert "device" in errors

    def test_validate_invalid_language(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["language"] = "fr"
        errors = mgr.validate(bad)
        assert "language" in errors

    def test_validate_invalid_auto_start(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["auto_start"] = "yes"
        errors = mgr.validate(bad)
        assert "auto_start" in errors

    def test_validate_invalid_hotkey(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["hotkey"] = {"keys": ["A"]}  # Single non-function key
        errors = mgr.validate(bad)
        assert "hotkey.keys" in errors

    def test_validate_remote_asr_config(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["model"] = "remote_glm_low_volume"
        config["remote_asr"] = {
            "endpoint": "https://asr.example.test/v1/audio/transcriptions",
            "api_token": "secret",
            "timeout_seconds": 5.0,
            "fallback_model": "fun_asr_nano",
        }
        assert mgr.validate(config) == []

        config["remote_asr"]["endpoint"] = "not-a-url"
        assert "remote_asr.endpoint" in mgr.validate(config)

    def test_validate_text_correction_config(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["text_correction"] = {
            "enabled": True,
            "backend": "qwen",
            "model": "Qwen/Qwen3-0.6B",
            "device": "auto",
            "max_chars": 120,
            "max_new_tokens": 80,
            "local_files_only": True,
            "min_free_vram_mb": 1200,
        }
        assert mgr.validate(config) == []

        config["text_correction"]["backend"] = "remote"
        assert "text_correction.backend" in mgr.validate(config)

    def test_validate_single_function_key_hotkey(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        good = copy.deepcopy(DEFAULT_CONFIG)
        good["hotkey"] = {"keys": ["F8"]}
        errors = mgr.validate(good)
        assert "hotkey.keys" not in errors

    def test_validate_history_max_entries_boundary(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)

        bad["history_max_entries"] = 0
        errors = mgr.validate(bad)
        assert "history_max_entries" in errors

        bad["history_max_entries"] = 501
        errors = mgr.validate(bad)
        assert "history_max_entries" in errors

        bad["history_max_entries"] = 1
        errors = mgr.validate(bad)
        assert "history_max_entries" not in errors

        bad["history_max_entries"] = 500
        errors = mgr.validate(bad)
        assert "history_max_entries" not in errors

    def test_validate_empty_audio_input(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        bad = copy.deepcopy(DEFAULT_CONFIG)
        bad["audio_input"] = ""
        errors = mgr.validate(bad)
        assert "audio_input" in errors

    def test_atomic_write_survives_read(self, tmp_path):
        """Verify that the file written is valid JSON after save."""
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.save(DEFAULT_CONFIG)

        with open(mgr.config_file, "r") as f:
            data = json.load(f)
        assert data["model"] == DEFAULT_CONFIG["model"]

    def test_load_merges_new_keys(self, tmp_path):
        """Config file missing a key should get the default value."""
        config_file = os.path.join(str(tmp_path), "config.json")
        partial = {"model": "tiny", "device": "cpu", "audio_input": "auto",
                    "hotkey": {"keys": ["Ctrl", "Alt", "V"]},
                    "language": "en", "auto_start": False}
        # Missing: history_max_entries
        with open(config_file, "w") as f:
            json.dump(partial, f)

        mgr = ConfigManager(config_file=config_file)
        config = mgr.load()
        assert "history_max_entries" in config
        assert config["history_max_entries"] == DEFAULT_CONFIG["history_max_entries"]

    def test_get_default_config_returns_deep_copy(self):
        c1 = ConfigManager.get_default_config()
        c2 = ConfigManager.get_default_config()
        c1["model"] = "medium"
        assert c2["model"] == DEFAULT_CONFIG["model"]


# ===================================================================
# 2. HistoryManager tests
# ===================================================================

from bytecli.service.history_manager import HistoryManager


class TestHistoryManager:
    """Unit tests for HistoryManager add/get/persist cycle."""

    def _make_manager(self, tmp_path: str, max_entries: int = 50) -> HistoryManager:
        history_file = os.path.join(tmp_path, "history.json")
        return HistoryManager(history_file=history_file, max_entries=max_entries)

    def test_add_and_get(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.add("hello world", "tiny", 1500)
        entries = mgr.get_recent(10)
        assert len(entries) == 1
        assert entries[0]["text"] == "hello world"
        assert entries[0]["model"] == "tiny"
        assert entries[0]["duration_ms"] == 1500

    def test_fifo_eviction(self, tmp_path):
        mgr = self._make_manager(str(tmp_path), max_entries=3)
        mgr.load()
        for i in range(5):
            mgr.add(f"entry_{i}", "small", 100)
        entries = mgr.entries
        assert len(entries) == 3
        # Oldest entries should be evicted.
        texts = [e["text"] for e in entries]
        assert "entry_0" not in texts
        assert "entry_1" not in texts
        assert "entry_4" in texts

    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path)
        mgr = self._make_manager(path)
        mgr.load()
        mgr.add("test1", "small", 200)
        mgr.add("test2", "medium", 300)

        mgr2 = self._make_manager(path)
        mgr2.load()
        assert len(mgr2.entries) == 2

    def test_corrupt_file_recovery(self, tmp_path):
        history_file = os.path.join(str(tmp_path), "history.json")
        with open(history_file, "w") as f:
            f.write("CORRUPT DATA!!!")

        mgr = HistoryManager(history_file=history_file)
        mgr.load()
        assert len(mgr.entries) == 0
        assert os.path.isfile(history_file + ".bak")

    def test_empty_history(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        assert mgr.entries == []
        assert mgr.get_recent(5) == []
        assert mgr.get_all() == []

    def test_get_all_format(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.add("hello", "tiny", 100)
        result = mgr.get_all()
        assert len(result) == 1
        text, timestamp, entry_id = result[0]
        assert text == "hello"
        assert isinstance(timestamp, str)
        assert isinstance(entry_id, str)

    def test_get_recent_order(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.add("first", "tiny", 100)
        mgr.add("second", "tiny", 100)
        mgr.add("third", "tiny", 100)
        recent = mgr.get_recent(2)
        assert len(recent) == 2
        assert recent[0]["text"] == "third"  # newest first
        assert recent[1]["text"] == "second"

    def test_max_entries_setter(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.max_entries = 0
        assert mgr.max_entries == 1  # Clamped to minimum
        mgr.max_entries = 999
        assert mgr.max_entries == 500  # Clamped to maximum
        mgr.max_entries = 100
        assert mgr.max_entries == 100

    def test_entries_returns_deep_copy(self, tmp_path):
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.add("test", "tiny", 100)
        entries = mgr.entries
        entries[0]["text"] = "MODIFIED"
        assert mgr.entries[0]["text"] == "test"

    def test_atomic_write(self, tmp_path):
        """Verify the history file is valid JSON after persist."""
        mgr = self._make_manager(str(tmp_path))
        mgr.load()
        mgr.add("test", "tiny", 100)

        history_file = os.path.join(str(tmp_path), "history.json")
        with open(history_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1


# ===================================================================
# 3. ServiceStateMachine tests
# ===================================================================

from bytecli.service.state_machine import (
    ServiceStateMachine,
    ServiceState,
    ServiceEvent,
)


class TestServiceStateMachine:
    """Unit tests for the service lifecycle state machine."""

    def test_initial_state(self):
        sm = ServiceStateMachine()
        assert sm.state is ServiceState.STOPPED

    def test_valid_start_sequence(self):
        sm = ServiceStateMachine()
        assert sm.dispatch(ServiceEvent.EVT_START) is True
        assert sm.state is ServiceState.STARTING
        assert sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS) is True
        assert sm.state is ServiceState.RUNNING

    def test_stop_sequence(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        assert sm.dispatch(ServiceEvent.EVT_STOP) is True
        assert sm.state is ServiceState.STOPPING
        assert sm.dispatch(ServiceEvent.EVT_SHUTDOWN_DONE) is True
        assert sm.state is ServiceState.STOPPED

    def test_restart_sequence(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        assert sm.dispatch(ServiceEvent.EVT_RESTART) is True
        assert sm.state is ServiceState.RESTARTING
        assert sm.dispatch(ServiceEvent.EVT_SHUTDOWN_DONE) is True
        assert sm.state is ServiceState.STARTING

    def test_init_fail_goes_to_failed(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        assert sm.dispatch(ServiceEvent.EVT_INIT_FAIL) is True
        assert sm.state is ServiceState.FAILED

    def test_crash_from_running(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        assert sm.dispatch(ServiceEvent.EVT_CRASH) is True
        assert sm.state is ServiceState.FAILED

    def test_restart_from_failed(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_FAIL)
        assert sm.state is ServiceState.FAILED
        assert sm.dispatch(ServiceEvent.EVT_RESTART) is True
        assert sm.state is ServiceState.RESTARTING

    def test_illegal_transition_returns_false(self):
        sm = ServiceStateMachine()
        # Cannot stop from STOPPED.
        assert sm.dispatch(ServiceEvent.EVT_STOP) is False
        assert sm.state is ServiceState.STOPPED

    def test_illegal_transition_preserves_state(self):
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        # Cannot start from RUNNING.
        assert sm.dispatch(ServiceEvent.EVT_START) is False
        assert sm.state is ServiceState.RUNNING

    def test_callback_fires_on_transition(self):
        records = []
        sm = ServiceStateMachine(
            on_state_change=lambda old, new: records.append((old, new))
        )
        sm.dispatch(ServiceEvent.EVT_START)
        assert len(records) == 1
        assert records[0] == (ServiceState.STOPPED, ServiceState.STARTING)

    def test_callback_error_does_not_corrupt_state(self):
        def bad_callback(old, new):
            raise ValueError("boom")

        sm = ServiceStateMachine(on_state_change=bad_callback)
        sm.dispatch(ServiceEvent.EVT_START)
        assert sm.state is ServiceState.STARTING

    def test_all_transitions_from_table(self):
        """Verify every defined transition works."""
        from bytecli.service.state_machine import _TRANSITIONS

        for (start_state, event), end_state in _TRANSITIONS.items():
            sm = ServiceStateMachine()
            sm._state = start_state
            result = sm.dispatch(event)
            assert result is True, f"{start_state} + {event} should be valid"
            assert sm.state is end_state


# ===================================================================
# 4. I18nManager tests
# ===================================================================

from bytecli.i18n.manager import I18nManager


class TestI18nManager:
    """Unit tests for the I18n translation manager."""

    def test_default_language_is_english(self):
        mgr = I18nManager()
        assert mgr.current_language == "en"

    def test_missing_key_returns_fallback(self):
        mgr = I18nManager()
        result = mgr.t("nonexistent.key", fallback="My Fallback")
        assert result == "My Fallback"

    def test_missing_key_returns_key_when_no_fallback(self):
        mgr = I18nManager()
        result = mgr.t("nonexistent.key")
        assert result == "nonexistent.key"

    def test_variable_interpolation(self):
        mgr = I18nManager()
        # Manually set a string with interpolation.
        mgr._strings["test.hello"] = "Hello {name}!"
        result = mgr.t("test.hello", name="World")
        assert result == "Hello World!"

    def test_missing_interpolation_variable(self):
        mgr = I18nManager()
        mgr._strings["test.key"] = "Hello {missing_var}!"
        result = mgr.t("test.key")
        # Should return the template without crashing.
        assert result == "Hello {missing_var}!"

    def test_switch_language(self):
        mgr = I18nManager()
        mgr.switch("zh")
        assert mgr.current_language == "zh"

    def test_switch_to_same_language_is_noop(self):
        mgr = I18nManager()
        callback = MagicMock()
        mgr.on_language_changed(callback)
        mgr.switch("en")  # Already en
        callback.assert_not_called()

    def test_language_change_callback(self):
        mgr = I18nManager()
        callback = MagicMock()
        mgr.on_language_changed(callback)
        mgr.switch("zh")
        callback.assert_called_once_with("zh")

    def test_remove_callback(self):
        mgr = I18nManager()
        callback = MagicMock()
        mgr.on_language_changed(callback)
        mgr.remove_language_changed(callback)
        mgr.switch("zh")
        callback.assert_not_called()

    def test_remove_nonexistent_callback(self):
        mgr = I18nManager()
        mgr.remove_language_changed(lambda x: None)  # Should not raise.

    def test_unsupported_language_falls_back(self):
        mgr = I18nManager()
        mgr.load("fr")  # Not supported
        assert mgr.current_language == "en"

    def test_corrupt_locale_file(self, tmp_path):
        mgr = I18nManager()
        # Temporarily override the locale dir.
        import bytecli.i18n.manager as m
        old_dir = m._LOCALE_DIR
        m._LOCALE_DIR = str(tmp_path)

        corrupt_file = os.path.join(str(tmp_path), "en.json")
        with open(corrupt_file, "w") as f:
            f.write("NOT JSON")

        mgr.load("en")
        # Should have empty strings but not crash.
        m._LOCALE_DIR = old_dir

    def test_duplicate_callback_registration(self):
        mgr = I18nManager()
        callback = MagicMock()
        mgr.on_language_changed(callback)
        mgr.on_language_changed(callback)  # Duplicate
        mgr.switch("zh")
        # Should only be called once.
        callback.assert_called_once()

    def test_callback_error_does_not_prevent_others(self):
        mgr = I18nManager()

        def bad_callback(lang):
            raise RuntimeError("boom")

        good_callback = MagicMock()
        mgr.on_language_changed(bad_callback)
        mgr.on_language_changed(good_callback)
        mgr.switch("zh")
        good_callback.assert_called_once_with("zh")


# ===================================================================
# 5. PidManager tests
# ===================================================================

from bytecli.service.pid_manager import PidManager


class TestPidManager:
    """Unit tests for PID file management."""

    def test_write_and_check(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        PidManager.check_and_write(pid_file)
        assert os.path.isfile(pid_file)
        with open(pid_file) as f:
            assert int(f.read().strip()) == os.getpid()

    def test_cleanup_removes_own_pid(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        PidManager.cleanup(pid_file)
        assert not os.path.isfile(pid_file)

    def test_cleanup_preserves_other_pid(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        with open(pid_file, "w") as f:
            f.write("99999999")  # Very unlikely to be a real PID
        PidManager.cleanup(pid_file)
        assert os.path.isfile(pid_file)

    def test_stale_pid_detected(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        with open(pid_file, "w") as f:
            f.write("99999999")  # Dead PID
        # Should not raise (stale PID should be cleaned).
        PidManager.check_and_write(pid_file)

    def test_is_running_no_file(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "nonexistent.pid")
        assert PidManager.is_running(pid_file) is False

    def test_is_running_corrupt_file(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        with open(pid_file, "w") as f:
            f.write("not_a_number")
        assert PidManager.is_running(pid_file) is False

    def test_own_pid_not_considered_running(self, tmp_path):
        """os.execv restart scenario: same PID should not block."""
        pid_file = os.path.join(str(tmp_path), "test.pid")
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        assert PidManager.is_running(pid_file) is False

    def test_double_check_and_write_same_process(self, tmp_path):
        pid_file = os.path.join(str(tmp_path), "test.pid")
        PidManager.check_and_write(pid_file)
        # Same PID should be allowed (self-restart scenario).
        PidManager.check_and_write(pid_file)


# ===================================================================
# 6. ModelSwitcher tests
# ===================================================================

from bytecli.service.model_switcher import ModelSwitcher, ModelSwitchState


class TestModelSwitcher:
    """Unit tests for the async model switcher."""

    def _make_engine_mock(self):
        engine = MagicMock()
        engine.current_model = "small"
        engine.current_device = "cpu"
        engine.load_model = MagicMock()
        engine.unload_model = MagicMock()
        return engine

    def test_switch_model_success(self):
        engine = self._make_engine_mock()
        switcher = ModelSwitcher(engine)
        results = []

        def callback(state, msg):
            results.append((state, msg))

        assert switcher.switch_model("tiny", callback) is True

        # Wait for the background thread to finish.
        time.sleep(0.5)

        assert switcher.state is ModelSwitchState.IDLE
        states = [r[0] for r in results]
        assert "switching" in states
        assert "success" in states
        engine.unload_model.assert_called_once()
        engine.load_model.assert_called_once_with("tiny", "cpu")

    def test_switch_model_uses_gpu_when_current_device_unknown(self, monkeypatch):
        from bytecli.service import model_switcher as model_switcher_module

        engine = self._make_engine_mock()
        engine.current_device = None
        switcher = ModelSwitcher(engine)
        callback = MagicMock()
        monkeypatch.setattr(
            model_switcher_module.WhisperEngine,
            "is_cuda_available",
            staticmethod(lambda: True),
        )

        assert switcher.switch_model("tiny", callback) is True
        time.sleep(0.5)

        engine.load_model.assert_called_once_with("tiny", "gpu")

    def test_switch_model_debounce(self):
        engine = self._make_engine_mock()
        # Make load_model block so the first switch is still in progress.
        engine.load_model = MagicMock(side_effect=lambda *a: time.sleep(1))
        switcher = ModelSwitcher(engine)
        callback = MagicMock()

        assert switcher.switch_model("tiny", callback) is True
        assert switcher.switch_model("medium", callback) is False

        # Wait for the first to finish.
        time.sleep(1.5)

    def test_switch_model_failure_and_rollback(self):
        engine = self._make_engine_mock()
        engine.unload_model = MagicMock()
        call_count = [0]

        def load_side_effect(model, device):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA OOM")
            # Rollback load succeeds.

        engine.load_model = MagicMock(side_effect=load_side_effect)
        switcher = ModelSwitcher(engine)
        results = []

        def callback(state, msg):
            results.append((state, msg))

        switcher.switch_model("medium", callback)
        time.sleep(0.5)

        assert switcher.state is ModelSwitchState.IDLE
        states = [r[0] for r in results]
        assert "failed" in states

    def test_switch_device_success(self):
        engine = self._make_engine_mock()
        switcher = ModelSwitcher(engine)
        results = []

        def callback(state, msg):
            results.append((state, msg))

        assert switcher.switch_device("gpu", callback) is True
        time.sleep(0.5)

        states = [r[0] for r in results]
        assert "success" in states
        engine.unload_model.assert_called_once()
        engine.load_model.assert_called_once_with("small", "gpu")

    def test_double_finish_guard(self):
        """Timeout + worker race: both call _finish, should not double-fire."""
        engine = self._make_engine_mock()
        # Make load_model slow enough that we can test the guard.
        engine.load_model = MagicMock(side_effect=lambda *a: time.sleep(0.2))
        switcher = ModelSwitcher(engine)
        results = []

        def callback(state, msg):
            results.append((state, msg))

        switcher.switch_model("tiny", callback)

        # Manually call _finish to simulate a timeout racing with the worker.
        time.sleep(0.1)
        switcher._finish(callback, "failed", "timeout")

        time.sleep(0.5)

        # The worker's _finish should see IDLE and return early.
        # Only switching + failed should be in results (no double success).
        final_states = [r[0] for r in results if r[0] != "switching"]
        assert len(final_states) == 1  # Either failed or success, not both.


# ===================================================================
# 7. RecordingFSM tests
# ===================================================================

from bytecli.service.recording_fsm import RecordingFSM, RecordingState


class TestRecordingFSM:
    """Unit tests for the recording finite state machine."""

    def _make_fsm(self):
        audio = MagicMock()
        engine = MagicMock()
        history = MagicMock()
        sig_started = MagicMock()
        sig_stopped = MagicMock()
        config_mgr = MagicMock()
        config_mgr.config = {"audio_input": "auto"}

        # Make stop_recording return a numpy-like array.
        import numpy as np
        audio.stop_recording.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio.start_recording.return_value = None

        engine.transcribe.return_value = "hello world"
        engine.current_model = "small"

        fsm = RecordingFSM(
            audio_manager=audio,
            whisper_engine=engine,
            history_manager=history,
            dbus_recording_started_signal=sig_started,
            dbus_recording_stopped_signal=sig_stopped,
            config_manager=config_mgr,
        )
        return fsm, audio, engine, history, sig_started, sig_stopped

    def test_initial_state(self):
        fsm, *_ = self._make_fsm()
        assert fsm.state is RecordingState.IDLE

    def test_toggle_starts_recording(self):
        fsm, audio, *_ = self._make_fsm()
        fsm.on_hotkey_toggle()
        assert fsm.state is RecordingState.RECORDING
        audio.start_recording.assert_called_once()

    def test_toggle_during_recording_stops_and_transcribes(self):
        fsm, audio, engine, *_ = self._make_fsm()
        fsm.on_hotkey_toggle()  # Start
        assert fsm.state is RecordingState.RECORDING

        # Wait slightly longer than MIN_RECORDING_DURATION.
        time.sleep(0.4)
        fsm.on_hotkey_toggle()  # Stop

        # Should be TRANSCRIBING (submitted to thread pool).
        assert fsm.state is RecordingState.TRANSCRIBING

    def test_stop_recording_emits_transcription_started(self):
        import numpy as np

        audio = MagicMock()
        engine = MagicMock()
        history = MagicMock()
        sig_started = MagicMock()
        sig_stopped = MagicMock()
        sig_transcription_started = MagicMock()
        config_mgr = MagicMock()
        config_mgr.config = {"audio_input": "auto"}
        audio.stop_recording.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        engine.transcribe.return_value = "hello world"
        engine.current_model = "small"

        fsm = RecordingFSM(
            audio_manager=audio,
            whisper_engine=engine,
            history_manager=history,
            dbus_recording_started_signal=sig_started,
            dbus_recording_stopped_signal=sig_stopped,
            config_manager=config_mgr,
            dbus_transcription_started_signal=sig_transcription_started,
        )

        fsm.on_hotkey_toggle()
        time.sleep(0.4)
        fsm.on_hotkey_toggle()

        assert fsm.state is RecordingState.TRANSCRIBING
        sig_transcription_started.assert_called_once_with()
        sig_stopped.assert_not_called()
        fsm.shutdown()

    def test_short_press_discarded(self):
        fsm, audio, engine, *_ = self._make_fsm()
        fsm.on_hotkey_toggle()  # Start
        # Immediately toggle (< 0.3s).
        fsm.on_hotkey_toggle()  # Stop
        # Should go back to IDLE (too short).
        assert fsm.state is RecordingState.IDLE

    def test_toggle_during_transcribing_ignored(self):
        fsm, *_ = self._make_fsm()
        fsm._state = RecordingState.TRANSCRIBING
        fsm.on_hotkey_toggle()
        assert fsm.state is RecordingState.TRANSCRIBING

    def test_start_recording_failure(self):
        fsm, audio, *_ = self._make_fsm()
        audio.start_recording.side_effect = RuntimeError("No device")
        fsm.on_hotkey_toggle()
        assert fsm.state is RecordingState.IDLE

    def test_empty_audio_buffer(self):
        import numpy as np
        fsm, audio, engine, history, _, sig_stopped = self._make_fsm()
        audio.stop_recording.return_value = np.array([], dtype=np.float32)

        fsm.on_hotkey_toggle()  # Start
        time.sleep(0.4)
        fsm.on_hotkey_toggle()  # Stop

        assert fsm.state is RecordingState.IDLE
        sig_stopped.assert_called_once_with("")

    def test_shutdown(self):
        fsm, *_ = self._make_fsm()
        fsm.shutdown()
        # Should not raise.

    def test_recording_started_signal_failure(self):
        fsm, audio, engine, history, sig_started, sig_stopped = self._make_fsm()
        sig_started.side_effect = RuntimeError("D-Bus error")
        # Should not crash.
        fsm.on_hotkey_toggle()
        assert fsm.state is RecordingState.RECORDING


# ===================================================================
# 8. WhisperEngine tests (mocked)
# ===================================================================

class TestWhisperEngine:
    """Unit tests for WhisperEngine lifecycle (Whisper/torch mocked)."""

    def test_initial_state(self):
        from bytecli.service.whisper_engine import WhisperEngine
        engine = WhisperEngine()
        assert engine.is_loaded is False
        assert engine.current_model is None
        assert engine.current_device is None

    def test_transcribe_without_model_raises(self):
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        engine = WhisperEngine()
        with pytest.raises(RuntimeError, match="No Whisper model"):
            engine.transcribe(np.zeros(16000, dtype=np.float32))

    def test_unload_when_not_loaded(self):
        from bytecli.service.whisper_engine import WhisperEngine
        engine = WhisperEngine()
        engine.unload_model()  # Should not raise.

    def test_load_and_unload(self):
        from bytecli.service.whisper_engine import WhisperEngine
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        import sys
        sys.modules["whisper"] = mock_whisper
        try:
            engine = WhisperEngine()
            engine.load_model("tiny", "cpu")
            assert engine.is_loaded is True
            assert engine.current_model == "tiny"
            assert engine.current_device == "cpu"

            engine.unload_model()
            assert engine.is_loaded is False
            assert engine.current_model is None
        finally:
            del sys.modules["whisper"]

    def test_load_model_failure(self):
        from bytecli.service.whisper_engine import WhisperEngine
        mock_whisper = MagicMock()
        mock_whisper.load_model.side_effect = Exception("download failed")

        engine = WhisperEngine()
        import sys
        sys.modules["whisper"] = mock_whisper
        try:
            with pytest.raises(RuntimeError, match="Failed to load"):
                engine.load_model("tiny", "cpu")
        finally:
            del sys.modules["whisper"]

    def test_faster_whisper_profile_transcribes_and_records_metrics(self):
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import sys

        captured = {}
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [types.SimpleNamespace(text=" hello"), types.SimpleNamespace(text=" world")],
            object(),
        )

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            def transcribe(self, *args, **kwargs):
                captured["transcribe_kwargs"] = kwargs
                return mock_model.transcribe(*args, **kwargs)

        sys.modules["faster_whisper"] = types.SimpleNamespace(
            WhisperModel=FakeWhisperModel
        )
        try:
            engine = WhisperEngine()
            engine.load_model("fast", "cpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.1)

            assert text == "hello world"
            assert engine.current_model == "fast"
            assert captured["kwargs"]["compute_type"] == "int8"
            assert captured["transcribe_kwargs"]["beam_size"] == 1
            assert captured["transcribe_kwargs"]["word_timestamps"] is False
            assert "initial_prompt" not in captured["transcribe_kwargs"]
            assert engine.last_metrics["backend"] == "faster_whisper"
            assert engine.last_metrics["profile"] == "fast"
        finally:
            del sys.modules["faster_whisper"]

    def test_qwen_profile_transcribes_and_records_metrics(self):
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import os
        import sys

        captured = {}
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [types.SimpleNamespace(text="你好 Qwen")]

        class FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return mock_model

        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float16=object(),
            float32=object(),
        )
        old_torch = sys.modules.get("torch")
        old_qwen_asr = sys.modules.get("qwen_asr")
        old_qwen_context = os.environ.get("BYTECLI_QWEN_ASR_CONTEXT")
        sys.modules["torch"] = fake_torch
        sys.modules["qwen_asr"] = types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel)
        os.environ.pop("BYTECLI_QWEN_ASR_CONTEXT", None)
        try:
            engine = WhisperEngine()
            engine.load_model("experimental_qwen", "gpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.1)

            assert text == "你好 Qwen"
            assert captured["args"] == ("Qwen/Qwen3-ASR-0.6B",)
            assert captured["kwargs"]["dtype"] is fake_torch.bfloat16
            assert captured["kwargs"]["device_map"] == "cuda:0"
            assert captured["kwargs"]["max_inference_batch_size"] == 1
            assert captured["kwargs"]["max_new_tokens"] == 256
            mock_model.transcribe.assert_called_once()
            transcribe_kwargs = mock_model.transcribe.call_args.kwargs
            audio_arg, sample_rate = transcribe_kwargs["audio"]
            assert sample_rate == 16000
            assert audio_arg.dtype == np.float32
            assert "context" in transcribe_kwargs
            assert "API" in transcribe_kwargs["context"]
            assert "Python" in transcribe_kwargs["context"]
            assert "中文内容保持中文" in transcribe_kwargs["context"]
            assert "不要翻译成英文" in transcribe_kwargs["context"]
            assert "不要翻译成中文" in transcribe_kwargs["context"]
            assert transcribe_kwargs["language"] is None
            assert transcribe_kwargs["return_time_stamps"] is False
            assert engine.last_metrics["backend"] == "qwen_asr"
            assert engine.last_metrics["profile"] == "experimental_qwen"
        finally:
            if old_torch is not None:
                sys.modules["torch"] = old_torch
            else:
                del sys.modules["torch"]
            if old_qwen_asr is not None:
                sys.modules["qwen_asr"] = old_qwen_asr
            else:
                del sys.modules["qwen_asr"]
            if old_qwen_context is None:
                os.environ.pop("BYTECLI_QWEN_ASR_CONTEXT", None)
            else:
                os.environ["BYTECLI_QWEN_ASR_CONTEXT"] = old_qwen_context

    def test_fun_asr_nano_profile_transcribes_and_records_metrics(self):
        from bytecli.service import audio_preprocess
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import os
        import sys
        import tempfile

        captured = {}

        class FakeFunASRModel:
            def generate(self, *args, **kwargs):
                captured["generate_args"] = args
                captured["generate_kwargs"] = kwargs
                return [{"text": "你好 FunASR"}]

        class FakeAutoModel:
            def __init__(self, *args, **kwargs):
                captured["load_args"] = args
                captured["load_kwargs"] = kwargs

            def generate(self, *args, **kwargs):
                return FakeFunASRModel().generate(*args, **kwargs)

        old_funasr = sys.modules.get("funasr")
        old_detect = audio_preprocess.detect_speech_ratio
        old_model_dir = os.environ.get("BYTECLI_FUN_ASR_MODEL_DIR")
        old_language = os.environ.get("BYTECLI_FUN_ASR_LANGUAGE")
        sys.modules["funasr"] = types.SimpleNamespace(AutoModel=FakeAutoModel)
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        os.environ.pop("BYTECLI_FUN_ASR_LANGUAGE", None)
        try:
            with tempfile.TemporaryDirectory() as model_dir:
                os.environ["BYTECLI_FUN_ASR_MODEL_DIR"] = model_dir
                engine = WhisperEngine()
                engine.load_model("fun_asr_nano", "gpu")
                text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01)

            assert text == "你好 FunASR"
            assert captured["load_kwargs"]["model"] == model_dir
            assert captured["load_kwargs"]["trust_remote_code"] is True
            assert captured["load_kwargs"]["device"] == "cuda:0"
            assert captured["load_kwargs"]["hub"] == "hf"
            assert captured["load_kwargs"]["disable_update"] is True
            assert captured["load_kwargs"]["disable_pbar"] is True
            assert captured["generate_kwargs"]["batch_size"] == 1
            assert captured["generate_kwargs"]["language"] == "auto"
            assert captured["generate_kwargs"]["itn"] is True
            assert captured["generate_kwargs"]["disable_pbar"] is True
            assert captured["generate_kwargs"]["input"][0].endswith(".wav")
            assert engine.last_metrics["backend"] == "funasr_nano"
            assert engine.last_metrics["profile"] == "fun_asr_nano"
        finally:
            audio_preprocess.detect_speech_ratio = old_detect
            if old_model_dir is None:
                os.environ.pop("BYTECLI_FUN_ASR_MODEL_DIR", None)
            else:
                os.environ["BYTECLI_FUN_ASR_MODEL_DIR"] = old_model_dir
            if old_language is None:
                os.environ.pop("BYTECLI_FUN_ASR_LANGUAGE", None)
            else:
                os.environ["BYTECLI_FUN_ASR_LANGUAGE"] = old_language
            if old_funasr is not None:
                sys.modules["funasr"] = old_funasr
            else:
                del sys.modules["funasr"]

    def test_sherpa_sensevoice_profile_transcribes_and_records_metrics(self, monkeypatch):
        from bytecli.service import audio_preprocess
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import sys

        captured = {}

        class FakeStream:
            def __init__(self):
                self.result = types.SimpleNamespace(text="你好 SenseVoice")

            def accept_waveform(self, sample_rate, waveform):
                captured["sample_rate"] = sample_rate
                captured["waveform_dtype"] = waveform.dtype

        class FakeRecognizer:
            def create_stream(self):
                return FakeStream()

            def decode_stream(self, stream):
                captured["decoded"] = True

        class FakeOfflineRecognizer:
            @classmethod
            def from_sense_voice(cls, **kwargs):
                captured["load_kwargs"] = kwargs
                return FakeRecognizer()

        old_sherpa = sys.modules.get("sherpa_onnx")
        old_detect = audio_preprocess.detect_speech_ratio
        sys.modules["sherpa_onnx"] = types.SimpleNamespace(
            OfflineRecognizer=FakeOfflineRecognizer
        )
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        try:
            tmp_dir = tempfile.TemporaryDirectory()
            model_dir = tmp_dir.name
            open(os.path.join(model_dir, "model.int8.onnx"), "w").close()
            open(os.path.join(model_dir, "tokens.txt"), "w").close()
            monkeypatch.setenv("BYTECLI_SHERPA_SENSEVOICE_MODEL_DIR", model_dir)

            engine = WhisperEngine()
            engine.load_model("sherpa_sensevoice", "gpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01)

            assert text == "你好 SenseVoice"
            assert captured["load_kwargs"]["model"].endswith("model.int8.onnx")
            assert captured["load_kwargs"]["tokens"].endswith("tokens.txt")
            assert captured["load_kwargs"]["provider"] == "cpu"
            assert captured["load_kwargs"]["language"] == ""
            assert captured["load_kwargs"]["use_itn"] is True
            assert captured["sample_rate"] == 16000
            assert captured["waveform_dtype"] == np.float32
            assert captured["decoded"] is True
            assert engine.last_metrics["backend"] == "sherpa_sensevoice"
            assert engine.last_metrics["profile"] == "sherpa_sensevoice"
        finally:
            if "tmp_dir" in locals():
                tmp_dir.cleanup()
            audio_preprocess.detect_speech_ratio = old_detect
            if old_sherpa is not None:
                sys.modules["sherpa_onnx"] = old_sherpa
            else:
                del sys.modules["sherpa_onnx"]

    def test_sherpa_funasr_nano_profile_transcribes_and_records_metrics(self, monkeypatch):
        from bytecli.service import audio_preprocess
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import sys

        captured = {}

        class FakeStream:
            result = '{"text":"你好 Sherpa FunASR"}'

            def accept_waveform(self, sample_rate, waveform):
                captured["sample_rate"] = sample_rate
                captured["waveform_dtype"] = waveform.dtype

        class FakeRecognizer:
            def create_stream(self):
                return FakeStream()

            def decode_stream(self, stream):
                captured["decoded"] = True

        class FakeOfflineRecognizer:
            @classmethod
            def from_funasr_nano(cls, **kwargs):
                captured["load_kwargs"] = kwargs
                return FakeRecognizer()

        old_sherpa = sys.modules.get("sherpa_onnx")
        old_detect = audio_preprocess.detect_speech_ratio
        sys.modules["sherpa_onnx"] = types.SimpleNamespace(
            OfflineRecognizer=FakeOfflineRecognizer
        )
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        try:
            tmp_dir = tempfile.TemporaryDirectory()
            model_dir = tmp_dir.name
            os.makedirs(os.path.join(model_dir, "Qwen3-0.6B"))
            open(os.path.join(model_dir, "encoder_adaptor.int8.onnx"), "w").close()
            open(os.path.join(model_dir, "llm.int8.onnx"), "w").close()
            open(os.path.join(model_dir, "embedding.int8.onnx"), "w").close()
            monkeypatch.setenv("BYTECLI_SHERPA_FUNASR_NANO_MODEL_DIR", model_dir)

            engine = WhisperEngine()
            engine.load_model("sherpa_funasr_nano", "gpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01)

            assert text == "你好 Sherpa FunASR"
            assert captured["load_kwargs"]["encoder_adaptor"].endswith(
                "encoder_adaptor.int8.onnx"
            )
            assert captured["load_kwargs"]["llm"].endswith("llm.int8.onnx")
            assert captured["load_kwargs"]["embedding"].endswith("embedding.int8.onnx")
            assert captured["load_kwargs"]["tokenizer"].endswith("Qwen3-0.6B")
            assert captured["load_kwargs"]["provider"] == "cpu"
            assert captured["load_kwargs"]["user_prompt"] == "语音转写:"
            assert captured["load_kwargs"]["max_new_tokens"] == 256
            assert captured["sample_rate"] == 16000
            assert captured["waveform_dtype"] == np.float32
            assert captured["decoded"] is True
            assert engine.last_metrics["backend"] == "sherpa_funasr_nano"
            assert engine.last_metrics["profile"] == "sherpa_funasr_nano"
        finally:
            if "tmp_dir" in locals():
                tmp_dir.cleanup()
            audio_preprocess.detect_speech_ratio = old_detect
            if old_sherpa is not None:
                sys.modules["sherpa_onnx"] = old_sherpa
            else:
                del sys.modules["sherpa_onnx"]

    def test_remote_asr_profile_posts_audio_and_records_metrics(self, monkeypatch):
        from bytecli.service import audio_preprocess
        from bytecli.service.whisper_engine import WhisperEngine
        import bytecli.service.whisper_engine as whisper_engine
        import numpy as np

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "text": "远端 GLM",
                        "backend": "glm_asr",
                        "inference_seconds": 1.2,
                        "total_seconds": 1.3,
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            captured["body"] = request.data
            captured["headers"] = dict(request.header_items())
            return FakeResponse()

        old_detect = audio_preprocess.detect_speech_ratio
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        monkeypatch.setenv(
            "BYTECLI_REMOTE_ASR_ENDPOINT",
            "https://asr.example.test/v1/audio/transcriptions",
        )
        monkeypatch.setenv("BYTECLI_REMOTE_ASR_TOKEN", "test-token")
        monkeypatch.setenv("BYTECLI_REMOTE_ASR_TIMEOUT", "7.5")
        monkeypatch.setattr(whisper_engine.urllib.request, "urlopen", fake_urlopen)
        try:
            engine = WhisperEngine()
            engine.load_model("remote_glm_low_volume", "gpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.05)
        finally:
            audio_preprocess.detect_speech_ratio = old_detect

        assert text == "远端 GLM"
        assert captured["timeout"] == 7.5
        assert captured["request"].full_url == (
            "https://asr.example.test/v1/audio/transcriptions"
        )
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert b"backend" in captured["body"]
        assert b"glm_asr" in captured["body"]
        assert b"recording.wav" in captured["body"]
        assert engine.last_metrics["backend"] == "remote_asr"
        assert engine.last_metrics["profile"] == "remote_glm_low_volume"

    def test_qwen_hallucination_phrase_is_blocked(self):
        from bytecli.service import audio_preprocess
        from bytecli.service.whisper_engine import WhisperEngine
        import numpy as np
        import sys

        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            types.SimpleNamespace(text="请点准确认识英语和英语的语音转录。")
        ]

        class FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return mock_model

        fake_torch = types.SimpleNamespace(
            bfloat16=object(),
            float16=object(),
            float32=object(),
        )
        old_torch = sys.modules.get("torch")
        old_qwen_asr = sys.modules.get("qwen_asr")
        sys.modules["torch"] = fake_torch
        sys.modules["qwen_asr"] = types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel)
        old_detect = audio_preprocess.detect_speech_ratio
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        try:
            engine = WhisperEngine()
            engine.load_model("experimental_qwen", "gpu")
            text = engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01)

            assert text == ""
            assert engine.last_metrics["hallucination_blocked"] is True
            assert engine.last_metrics["validation_reason"] == "hallucination_pattern"
        finally:
            audio_preprocess.detect_speech_ratio = old_detect
            if old_torch is not None:
                sys.modules["torch"] = old_torch
            else:
                del sys.modules["torch"]
            if old_qwen_asr is not None:
                sys.modules["qwen_asr"] = old_qwen_asr
            else:
                del sys.modules["qwen_asr"]

    def test_qwen_missing_dependency_message(self):
        from bytecli.service.whisper_engine import WhisperEngine
        import sys

        old_qwen_asr = sys.modules.get("qwen_asr")
        sys.modules["qwen_asr"] = None
        engine = WhisperEngine()
        try:
            with pytest.raises(RuntimeError, match="qwen-asr could not be imported"):
                engine.load_model("experimental_qwen", "cpu")
        finally:
            if old_qwen_asr is not None:
                sys.modules["qwen_asr"] = old_qwen_asr
            else:
                del sys.modules["qwen_asr"]

    def test_collapse_repeats(self):
        from bytecli.service.whisper_engine import _collapse_repeats
        assert _collapse_repeats("我我我我我我我我") == "我我我"
        assert _collapse_repeats("hello") == "hello"
        assert _collapse_repeats("") == ""
        assert _collapse_repeats("aaa") == "aaa"  # Exactly 3 is OK
        assert _collapse_repeats("aaaa") == "aaa"  # 4 collapses to 3

    def test_transcript_cleanup_removes_fillers_and_corrects_common_errors(self):
        from bytecli.service.transcript_cleanup import cleanup_transcript

        result = cleanup_transcript("嗯就是, 帮我重新再打一个炮, 非常小的生意")

        assert result.text == "帮我重新打一个包，非常小的声音"
        assert result.changed is True
        assert result.elapsed_ms < 5.0

    def test_transcript_cleanup_preserves_intentional_middle_words(self):
        from bytecli.service.transcript_cleanup import cleanup_transcript

        result = cleanup_transcript("先打开设置，然后切换到 Qwen")

        assert result.text == "先打开设置，然后切换到 Qwen"

    def test_transcript_cleanup_uses_qwen_when_ready(self, monkeypatch):
        from bytecli.service import transcript_cleanup
        from bytecli.service.transcript_cleanup import (
            TextCorrectionSettings,
            cleanup_transcript,
        )

        class FakeCorrector:
            def correct_if_ready(self, text, settings):
                return "帮我重新打一个包"

        monkeypatch.setattr(transcript_cleanup, "_CORRECTOR", FakeCorrector())
        settings = TextCorrectionSettings(backend="qwen")

        result = cleanup_transcript("帮我重新打一个炮", settings=settings)

        assert result.text == "帮我重新打一个包"
        assert result.backend == "qwen"

    def test_low_volume_audio_preprocessor_normalizes_without_skipping(self):
        from bytecli.service import audio_preprocess
        from bytecli.service.audio_preprocess import AudioPreprocessor
        import numpy as np

        t = np.linspace(0, 1, 16000, endpoint=False)
        quiet = (np.sin(2 * np.pi * 220 * t) * 0.001).astype(np.float32)
        old_detect = audio_preprocess.detect_speech_ratio
        audio_preprocess.detect_speech_ratio = lambda audio: (None, "energy")
        try:
            result = AudioPreprocessor("vad_norm").process(quiet)
        finally:
            audio_preprocess.detect_speech_ratio = old_detect

        assert result.skipped is False
        assert result.gain_db > 10
        assert result.output.peak <= 1.0
        assert result.output.rms > result.input.rms

    def test_silent_audio_preprocessor_skips(self):
        from bytecli.service.audio_preprocess import AudioPreprocessor
        import numpy as np

        result = AudioPreprocessor("vad_norm").process(np.zeros(16000, dtype=np.float32))

        assert result.skipped is True
        assert result.skip_reason == "no_speech"

    def test_eval_error_metrics(self):
        from bytecli.eval.asr_eval import _cer, _wer

        assert _cer("你好", "你号") == 0.5
        assert _wer("", "") == 0.0


# ===================================================================
# 9. D-Bus service input validation tests
# ===================================================================

class TestDBusServiceValidation:
    """Test that D-Bus SwitchModel/SwitchDevice validate inputs."""

    def _make_service(self):
        bus_name = MagicMock()
        config = MagicMock()
        state = MagicMock()
        engine = MagicMock()
        audio = MagicMock()
        hotkey = MagicMock()
        history = MagicMock()
        switcher = MagicMock()
        switcher.switch_model.return_value = True
        switcher.switch_device.return_value = True

        # We need to mock dbus.service.Object to avoid needing a real bus.
        with patch("dbus.service.Object.__init__", return_value=None):
            from bytecli.service.dbus_service import ByteCLIDBusService
            svc = ByteCLIDBusService(
                bus_name=bus_name,
                config_manager=config,
                state_machine=state,
                whisper_engine=engine,
                audio_manager=audio,
                hotkey_manager=hotkey,
                history_manager=history,
                model_switcher=switcher,
            )
        return svc, switcher

    def test_switch_model_valid(self):
        svc, switcher = self._make_service()
        result = svc.SwitchModel("tiny")
        assert result is True
        switcher.switch_model.assert_called_once()

    def test_switch_model_invalid(self):
        svc, switcher = self._make_service()
        result = svc.SwitchModel("huge")
        assert result is False
        switcher.switch_model.assert_not_called()

    def test_switch_device_valid_gpu(self):
        svc, switcher = self._make_service()
        result = svc.SwitchDevice("gpu")
        assert result is True

    def test_switch_device_valid_cpu(self):
        svc, switcher = self._make_service()
        result = svc.SwitchDevice("cpu")
        assert result is True

    def test_switch_device_invalid(self):
        svc, switcher = self._make_service()
        result = svc.SwitchDevice("tpu")
        assert result is False
        switcher.switch_device.assert_not_called()


# ===================================================================
# 10. Corner case tests
# ===================================================================

class TestCornerCases:
    """Tests for edge cases and error scenarios."""

    def test_concurrent_model_switches(self):
        """Fire 3 switches simultaneously — only the first should proceed."""
        engine = MagicMock()
        engine.current_model = "small"
        engine.current_device = "cpu"
        engine.load_model = MagicMock(side_effect=lambda *a: time.sleep(0.5))
        engine.unload_model = MagicMock()

        switcher = ModelSwitcher(engine)
        callback = MagicMock()

        results = []
        results.append(switcher.switch_model("tiny", callback))
        results.append(switcher.switch_model("medium", callback))
        results.append(switcher.switch_model("small", callback))

        assert results[0] is True
        assert results[1] is False
        assert results[2] is False

        time.sleep(1.0)

    def test_config_file_deleted_mid_run(self, tmp_path):
        """Deleting the config file should not crash -- re-save works."""
        config_file = os.path.join(str(tmp_path), "config.json")
        mgr = ConfigManager(config_file=config_file)
        mgr.load()

        # Delete the file.
        os.unlink(config_file)
        assert not os.path.isfile(config_file)

        # Save should recreate it.
        mgr.save(DEFAULT_CONFIG)
        assert os.path.isfile(config_file)

    def test_history_corruption_recovery(self, tmp_path):
        """Corrupt history file should be backed up and cleared."""
        history_file = os.path.join(str(tmp_path), "history.json")

        # Write some valid entries first.
        mgr = HistoryManager(history_file=history_file)
        mgr.load()
        mgr.add("test1", "tiny", 100)
        mgr.add("test2", "small", 200)
        assert len(mgr.entries) == 2

        # Corrupt the file.
        with open(history_file, "w") as f:
            f.write("CORRUPTED!!!!")

        # Reload — should detect corruption and reset.
        mgr2 = HistoryManager(history_file=history_file)
        mgr2.load()
        assert len(mgr2.entries) == 0
        assert os.path.isfile(history_file + ".bak")

    def test_rapid_fsm_toggles(self):
        """10 rapid toggles should not crash the FSM."""
        audio = MagicMock()
        engine = MagicMock()
        history = MagicMock()
        sig_started = MagicMock()
        sig_stopped = MagicMock()
        config_mgr = MagicMock()
        config_mgr.config = {"audio_input": "auto"}

        import numpy as np
        audio.stop_recording.return_value = np.zeros(100, dtype=np.float32)
        engine.transcribe.return_value = "test"
        engine.current_model = "small"

        fsm = RecordingFSM(
            audio_manager=audio,
            whisper_engine=engine,
            history_manager=history,
            dbus_recording_started_signal=sig_started,
            dbus_recording_stopped_signal=sig_stopped,
            config_manager=config_mgr,
        )

        for _ in range(10):
            fsm.on_hotkey_toggle()

        # Should still be in a valid state.
        assert fsm.state in (
            RecordingState.IDLE,
            RecordingState.RECORDING,
            RecordingState.TRANSCRIBING,
        )
        fsm.shutdown()

    def test_state_machine_full_cycle(self):
        """Run through STOPPED->STARTING->RUNNING->STOPPING->STOPPED."""
        sm = ServiceStateMachine()
        assert sm.state is ServiceState.STOPPED

        sm.dispatch(ServiceEvent.EVT_START)
        assert sm.state is ServiceState.STARTING

        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        assert sm.state is ServiceState.RUNNING

        sm.dispatch(ServiceEvent.EVT_STOP)
        assert sm.state is ServiceState.STOPPING

        sm.dispatch(ServiceEvent.EVT_SHUTDOWN_DONE)
        assert sm.state is ServiceState.STOPPED

    def test_state_machine_crash_restart_cycle(self):
        """RUNNING -> FAILED -> RESTARTING -> STARTING."""
        sm = ServiceStateMachine()
        sm.dispatch(ServiceEvent.EVT_START)
        sm.dispatch(ServiceEvent.EVT_INIT_SUCCESS)
        sm.dispatch(ServiceEvent.EVT_CRASH)
        assert sm.state is ServiceState.FAILED

        sm.dispatch(ServiceEvent.EVT_RESTART)
        assert sm.state is ServiceState.RESTARTING

        sm.dispatch(ServiceEvent.EVT_SHUTDOWN_DONE)
        assert sm.state is ServiceState.STARTING

    def test_config_validation_all_wrong_types(self, tmp_path):
        """Every field with a wrong type should be caught."""
        mgr = ConfigManager(config_file=os.path.join(str(tmp_path), "c.json"))
        bad = {
            "model": 123,
            "device": None,
            "audio_input": 0,
            "hotkey": "not a dict",
            "language": True,
            "auto_start": "yes",
            "history_max_entries": "fifty",
        }
        errors = mgr.validate(bad)
        assert len(errors) >= 6  # All fields should be invalid

    def test_i18n_fallback_parameter(self):
        """Verify that fallback= is properly used by i18n.t()."""
        from bytecli.i18n import i18n
        # Use a key that definitely doesn't exist.
        result = i18n.t("this.key.absolutely.does.not.exist", fallback="Fallback Text")
        assert result == "Fallback Text"

    def test_history_add_preserves_entry_fields(self, tmp_path):
        """Each history entry should have id, text, timestamp, model, duration_ms."""
        mgr = HistoryManager(
            history_file=os.path.join(str(tmp_path), "h.json"),
            max_entries=10,
        )
        mgr.load()
        mgr.add("test text", "medium", 5000)

        entry = mgr.entries[0]
        assert "id" in entry
        assert "text" in entry
        assert "timestamp" in entry
        assert "model" in entry
        assert "duration_ms" in entry
        # Validate UUID format.
        uuid.UUID(entry["id"])  # Should not raise.

    def test_model_switcher_rollback_on_load_failure(self):
        """If the new model fails to load, rollback should reload the old one."""
        engine = MagicMock()
        engine.current_model = "small"
        engine.current_device = "cpu"
        call_count = [0]

        def load_side_effect(model, device):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("load failed")

        engine.load_model = MagicMock(side_effect=load_side_effect)
        switcher = ModelSwitcher(engine)
        results = []

        switcher.switch_model("tiny", lambda s, m: results.append(s))
        time.sleep(0.5)

        # Should have attempted rollback (2 load_model calls: new + rollback).
        assert engine.load_model.call_count == 2
        # Rollback call should be with the old model.
        rollback_call = engine.load_model.call_args_list[1]
        assert rollback_call[0] == ("small", "cpu")

    def test_pid_file_permissions(self, tmp_path):
        """PID file should be writable by current user."""
        pid_file = os.path.join(str(tmp_path), "test.pid")
        PidManager.check_and_write(pid_file)
        assert os.access(pid_file, os.R_OK | os.W_OK)
