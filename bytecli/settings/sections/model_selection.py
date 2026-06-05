"""
ModelSelectionSection -- ASR inference profile picker.

Displays three RadioOption rows and handles the asynchronous model
switching workflow, including spinner, checkmark, failure states and
auto-revert on error.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import GLib, Gtk

from bytecli.constants import INFERENCE_PROFILES, VISIBLE_INFERENCE_PROFILES
from bytecli.i18n import i18n
from bytecli.shared.dbus_client import DBusClient
from bytecli.settings.widgets.section_card import SectionCard
from bytecli.settings.widgets.radio_option import RadioOption

logger = logging.getLogger(__name__)

# Profile keys in display order.
_MODEL_ORDER = list(VISIBLE_INFERENCE_PROFILES)


class ModelSelectionSection(Gtk.Box):
    """Radio group for ASR profile selection."""

    def __init__(
        self,
        dbus_client: DBusClient,
        config: dict[str, Any],
        on_changed: Callable[[], None],
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._dbus_client = dbus_client
        self._config = config
        self._on_changed = on_changed
        self._switching = False
        self._previous_model: Optional[str] = None
        self._pending_model: Optional[str] = None
        self._restore_timeout_id: Optional[int] = None
        self._switch_timeout_id: Optional[int] = None

        self._card = SectionCard(
            title=i18n.t("model.label", fallback="Model Selection")
        )

        # i18n keys for profile display names and descriptions.
        _model_i18n = {
            "fast": ("model.fast", "model.fast_desc"),
            "balanced": ("model.balanced", "model.balanced_desc"),
            "zh_fast": ("model.zh_fast", "model.zh_fast_desc"),
            "fun_asr_nano": ("model.fun_asr_nano", "model.fun_asr_nano_desc"),
            "experimental_qwen": ("model.qwen", "model.qwen_desc"),
            "remote_glm_low_volume": (
                "model.remote_glm",
                "model.remote_glm_desc",
            ),
            "remote_qwen_1_7b": ("model.remote_qwen", "model.remote_qwen_desc"),
            "remote_fun_asr_nano": (
                "model.remote_fun_asr",
                "model.remote_fun_asr_desc",
            ),
        }

        self._radios: dict[str, RadioOption] = {}
        for key in _MODEL_ORDER:
            meta = INFERENCE_PROFILES[key]
            name_key, desc_key = _model_i18n.get(key, ("", ""))
            display = i18n.t(name_key, fallback=str(meta["display_name"]))
            desc = i18n.t(desc_key, fallback=str(meta["description"]))
            is_recommended = key == (_MODEL_ORDER[0] if _MODEL_ORDER else "fast")

            radio = RadioOption(
                label_text=display,
                description_text=desc,
                on_clicked=lambda r, k=key: self._on_radio_clicked(k),
                highlight_description=is_recommended,
            )
            self._radios[key] = radio
            self._card.card_content.append(radio)

        self._performance_label = Gtk.Label()
        self._performance_label.add_css_class("text-muted")
        self._performance_label.add_css_class("text-sm")
        self._performance_label.set_halign(Gtk.Align.START)
        self._performance_label.set_wrap(True)
        self._card.card_content.append(self._performance_label)
        self._refresh_performance_label()

        self.append(self._card)

        # Set initial selection from config. Legacy OpenAI Whisper keys are
        # valid service fallbacks, but the settings UI presents profiles.
        initial_model = config.get("model", "fast")
        if initial_model not in self._radios:
            initial_model = "fast"
            self._config["model"] = initial_model
        self._apply_selection(initial_model)

        # Listen for model-switch progress from D-Bus.
        self._dbus_client.subscribe_signal(
            "ModelSwitchProgress", self._on_switch_progress
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @property
    def is_switching(self) -> bool:
        return self._switching

    def _apply_selection(self, model_key: str) -> None:
        if model_key not in self._radios:
            model_key = "fast"
        for key, radio in self._radios.items():
            radio.selected = key == model_key

    def _on_radio_clicked(self, key: str) -> None:
        if self._switching:
            return

        self._previous_model = self._config.get("model", "fast")
        self._pending_model = key
        self._apply_selection(key)

        # Begin the async model switch.
        self._switching = True
        self._set_switching_ui(key)
        self._on_changed()
        self._dbus_client.switch_model(key, callback=self._on_switch_result)
        # Safety timeout in case the service crashes mid-switch.
        self._switch_timeout_id = GLib.timeout_add(65000, self._on_switch_timeout)

    def _set_switching_ui(self, active_key: str) -> None:
        for key, radio in self._radios.items():
            if key == active_key:
                radio.show_spinner()
                radio.disabled = False
            else:
                radio.disabled = True

    def _restore_ui(self) -> bool:
        self._restore_timeout_id = None
        for radio in self._radios.values():
            radio.disabled = False
            radio._clear_status()
        return False

    # ------------------------------------------------------------------
    # D-Bus callbacks
    # ------------------------------------------------------------------

    def _on_switch_result(self, result) -> None:
        if not _dbus_bool(result):
            self._on_switch_failed()
        # A True result only means the service accepted the switch request.
        # Wait for ModelSwitchProgress("success") before saving the config.

    def _on_switch_timeout(self) -> bool:
        self._switch_timeout_id = None
        if self._switching:
            logger.warning("Model switch timed out in settings UI.")
            self._switching = False
            self._on_switch_failed()
        return False

    def _on_switch_progress(self, conn, sender, path, iface, signal_name, params) -> None:
        """Handle intermediate progress signals during model download."""
        if not self._switching or params is None:
            return

        unpacked = params.unpack()
        if not unpacked:
            return

        state = str(unpacked[0])
        message = str(unpacked[1]) if len(unpacked) > 1 else ""
        if state == "success":
            if self._switch_timeout_id is not None:
                GLib.source_remove(self._switch_timeout_id)
                self._switch_timeout_id = None
            self._switching = False
            self._on_switch_success()
        elif state == "failed":
            logger.warning("Model switch failed: %s", message)
            self._on_switch_failed()

    def _on_switch_success(self) -> None:
        if self._pending_model is not None:
            self._config["model"] = self._pending_model
            self._pending_model = None

        current = self._config.get("model", "fast")
        if current in self._radios:
            self._radios[current].show_checkmark(duration_ms=2000)
        self._refresh_performance_label()
        self._on_changed()
        self._schedule_restore()

    def _on_switch_failed(self) -> None:
        if self._switch_timeout_id is not None:
            GLib.source_remove(self._switch_timeout_id)
            self._switch_timeout_id = None
        self._switching = False

        failed_model = self._pending_model
        self._pending_model = None
        if failed_model in self._radios:
            self._radios[failed_model].show_x_mark()

        # Revert to previous model.
        if self._previous_model:
            self._config["model"] = self._previous_model
            self._apply_selection(self._previous_model)

        self._on_changed()
        self._schedule_restore()

    def _schedule_restore(self) -> None:
        if self._restore_timeout_id is not None:
            GLib.source_remove(self._restore_timeout_id)
        self._restore_timeout_id = GLib.timeout_add(2000, self._restore_ui)

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    def collect_config(self, config: dict) -> None:
        config["model"] = self._config.get("model", "fast")

    def apply_config(self, config: dict) -> None:
        self._config["model"] = config.get("model", "fast")
        self._pending_model = None
        self._apply_selection(self._config["model"])

    def refresh_labels(self) -> None:
        self._card.set_title(
            i18n.t("model.label", fallback="Model Selection")
        )
        model_i18n = {
            "fast": ("model.fast", "model.fast_desc"),
            "balanced": ("model.balanced", "model.balanced_desc"),
            "zh_fast": ("model.zh_fast", "model.zh_fast_desc"),
            "fun_asr_nano": ("model.fun_asr_nano", "model.fun_asr_nano_desc"),
            "experimental_qwen": ("model.qwen", "model.qwen_desc"),
            "remote_glm_low_volume": (
                "model.remote_glm",
                "model.remote_glm_desc",
            ),
            "remote_qwen_1_7b": ("model.remote_qwen", "model.remote_qwen_desc"),
            "remote_fun_asr_nano": (
                "model.remote_fun_asr",
                "model.remote_fun_asr_desc",
            ),
        }
        for key, radio in self._radios.items():
            meta = INFERENCE_PROFILES[key]
            name_key, desc_key = model_i18n.get(key, ("", ""))
            radio.label_text = i18n.t(name_key, fallback=str(meta["display_name"]))
            radio.description_text = i18n.t(desc_key, fallback=str(meta["description"]))
        self._refresh_performance_label()

    def _refresh_performance_label(self) -> None:
        metrics = self._dbus_client.get_last_performance()
        if not metrics:
            self._performance_label.set_text(
                i18n.t("model.last_perf_empty", fallback="Last transcription: none")
            )
            return

        backend = metrics.get("backend", "unknown")
        audio_s = metrics.get("audio_seconds", 0)
        infer_s = metrics.get("inference_seconds", 0)
        total_s = metrics.get("total_seconds", 0)
        text = i18n.t(
            "model.last_perf",
            backend=backend,
            audio=audio_s,
            infer=infer_s,
            total=total_s,
            fallback=(
                f"Last transcription: {backend}, audio {audio_s}s, "
                f"infer {infer_s}s, total {total_s}s"
            ),
        )
        self._performance_label.set_text(text)


def _dbus_bool(result) -> bool:
    if result is None:
        return False
    try:
        unpacked = result.unpack()
        if isinstance(unpacked, tuple) and unpacked:
            return bool(unpacked[0])
        return bool(unpacked)
    except Exception:
        return True
