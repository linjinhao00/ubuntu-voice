"""
IndicatorWindow -- floating pill-shaped status indicator.

Sits at the bottom-center of the primary monitor displaying the current
dictation state (idle / recording) with an elapsed-time counter.  On
hover the widget expands to reveal a History button that opens the
HistoryPanel popup.
"""

from __future__ import annotations

import logging
import math
import subprocess
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, GLib, Gtk, Pango

from bytecli.i18n import i18n
from bytecli.shared.dbus_client import DBusClient

logger = logging.getLogger(__name__)

# Margin from the bottom edge of the screen.
_BOTTOM_MARGIN = 92
_PILL_WIDTH = 160
_PILL_HEIGHT = 40
_WAVE_BAR_COUNT = 7
_POINTER_POLL_MS = 30

_X11_DISPLAY = None
_SCREEN_GEOMETRY: Optional[tuple[int, int, int, int]] = None


class IndicatorWindow(Gtk.Window):
    """Undecorated pill window pinned to the bottom-center of the screen."""

    def __init__(
        self,
        application: Gtk.Application,
        dbus_client: DBusClient,
    ) -> None:
        super().__init__(application=application)
        self._dbus_client = dbus_client
        self._recording = False
        self._downloading = False
        self._transcribing = False
        self._pulse_on = False
        self._audio_level = 0.0
        self._wave_phase = 0.0
        self._timer_seconds = 0
        self._timer_source_id: Optional[int] = None
        self._pulse_source_id: Optional[int] = None
        self._wave_source_id: Optional[int] = None
        self._leave_timeout_id: Optional[int] = None
        self._pointer_poll_source_id: Optional[int] = None
        self._history_panel = None
        self._saved_position = self._load_saved_position()
        self._drag_origin: tuple[int, int] | None = None
        self._drag_pointer_origin: tuple[int, int] | None = None
        self._indicator_geo: tuple[int, int, int, int] = (
            0,
            0,
            _PILL_WIDTH,
            _PILL_HEIGHT,
        )

        # Window chrome.
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_can_focus(True)
        self.set_focusable(True)
        self.set_title("ByteCLI Indicator")
        self.add_css_class("indicator-window")
        self.set_default_size(_PILL_WIDTH, _PILL_HEIGHT)

        # Build UI.
        self._build_ui()

        # Position after the window is realized so we can read monitor geometry.
        self.connect("realize", self._on_realize)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Root box with the pill CSS class.
        self._pill_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self._pill_box.add_css_class("indicator-pill")
        self._pill_box.set_size_request(_PILL_WIDTH, _PILL_HEIGHT)
        self._pill_box.set_margin_start(0)
        self._pill_box.set_margin_end(0)

        # --- Status dot ---------------------------------------------------
        self._dot = Gtk.DrawingArea()
        self._dot.set_size_request(7, 7)
        self._dot.set_valign(Gtk.Align.CENTER)
        self._dot.set_draw_func(self._draw_dot)
        self._pill_box.append(self._dot)

        # --- Live waveform (visible while recording) ---------------------
        self._wave = Gtk.DrawingArea()
        self._wave.add_css_class("indicator-waveform")
        self._wave.set_size_request(42, 18)
        self._wave.set_valign(Gtk.Align.CENTER)
        self._wave.set_draw_func(self._draw_waveform)
        self._wave.set_opacity(0.0)
        self._pill_box.append(self._wave)

        # --- Status text -------------------------------------------------
        self._status_label = Gtk.Label(label=i18n.t("indicator.idle", fallback="Idle"))
        self._status_label.add_css_class("mono")
        self._status_label.add_css_class("font-medium")
        self._status_label.set_valign(Gtk.Align.CENTER)
        self._status_label.set_size_request(36, -1)
        self._status_label.set_width_chars(4)
        self._status_label.set_max_width_chars(5)
        self._status_label.set_ellipsize(Pango.EllipsizeMode.END)
        _apply_font_size(self._status_label, 12)
        self._pill_box.append(self._status_label)

        # --- Timer label (visible only while recording) ------------------
        self._timer_label = Gtk.Label(label="00:00")
        self._timer_label.add_css_class("mono")
        self._timer_label.add_css_class("text-muted")
        self._timer_label.set_valign(Gtk.Align.CENTER)
        self._timer_label.set_size_request(34, -1)
        _apply_font_size(self._timer_label, 12)
        self._timer_label.set_opacity(0.0)
        self._pill_box.append(self._timer_label)

        # --- Separator (visible on hover) --------------------------------
        self._separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._separator.set_size_request(1, 16)
        self._separator.set_valign(Gtk.Align.CENTER)
        self._separator.set_visible(False)
        self._pill_box.append(self._separator)

        # --- History button (visible on hover) ---------------------------
        self._history_btn = Gtk.Button()
        self._history_btn.add_css_class("icon-btn")
        history_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        history_box.set_valign(Gtk.Align.CENTER)

        history_icon = Gtk.Image.new_from_icon_name("document-open-recent-symbolic")
        history_icon.set_pixel_size(14)
        history_box.append(history_icon)

        history_text = Gtk.Label(label=i18n.t("indicator.history", fallback="History"))
        history_text.add_css_class("text-sm")
        history_box.append(history_text)

        self._history_btn.set_child(history_box)
        self._history_btn.set_visible(False)
        self._history_btn.connect("clicked", self._on_history_clicked)
        self._pill_box.append(self._history_btn)

        # --- Hover detection ---------------------------------------------
        motion = Gtk.EventControllerMotion()
        motion.connect("enter", self._on_mouse_enter)
        motion.connect("leave", self._on_mouse_leave)
        self._pill_box.add_controller(motion)

        self.set_child(self._pill_box)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_dot(
        self,
        area: Gtk.DrawingArea,
        cr,
        width: int,
        height: int,
    ) -> None:
        """Draw a filled circle as the status dot."""
        if self._recording:
            cr.set_source_rgba(0.714, 1.0, 0.808, 1.0)  # #B6FFCE
        elif self._transcribing:
            alpha = 1.0 if self._pulse_on else 0.46
            cr.set_source_rgba(0.698, 0.698, 1.0, alpha)  # #B2B2FF
        elif self._downloading:
            cr.set_source_rgba(1.0, 0.518, 0.0, 1.0)    # #FF8400 (warning orange)
        else:
            cr.set_source_rgba(0.722, 0.725, 0.714, 1.0)  # #B8B9B6
        radius = min(width, height) / 2.0
        cr.arc(width / 2.0, height / 2.0, radius, 0, 2 * math.pi)
        cr.fill()

    def _draw_waveform(
        self,
        area: Gtk.DrawingArea,
        cr,
        width: int,
        height: int,
    ) -> None:
        """Draw voice-level bars while recording."""
        level = max(0.0, min(1.0, self._audio_level))
        bar_w = max(3.0, width / (_WAVE_BAR_COUNT * 2.6))
        gap = (width - (_WAVE_BAR_COUNT * bar_w)) / max(1, _WAVE_BAR_COUNT - 1)
        center_y = height / 2.0

        for i in range(_WAVE_BAR_COUNT):
            wave = 0.5 + 0.5 * math.sin(self._wave_phase + i * 0.78)
            shaped = 0.18 + level * (0.28 + 0.72 * wave)
            bar_h = max(4.0, min(height, shaped * height))
            x = i * (bar_w + gap)
            y = center_y - bar_h / 2.0

            if level > 0.08:
                cr.set_source_rgba(1.0, 0.518, 0.0, 0.92)
            else:
                cr.set_source_rgba(0.714, 1.0, 0.808, 0.42)
            _rounded_rect(cr, x, y, bar_w, bar_h, bar_w / 2.0)
            cr.fill()

    # ------------------------------------------------------------------
    # Positioning & X11 properties
    # ------------------------------------------------------------------

    def _on_realize(self, widget: Gtk.Widget) -> None:
        """Set X11 window hints and position the pill."""
        GLib.idle_add(self._apply_x11_properties)
        GLib.idle_add(self._position_on_screen)
        GLib.timeout_add(250, self._position_on_screen)
        GLib.timeout_add(900, self._position_on_screen)
        if self._pointer_poll_source_id is None:
            self._pointer_poll_source_id = GLib.timeout_add(
                _POINTER_POLL_MS,
                self._watch_pointer_drag,
            )

    def _apply_x11_properties(self) -> bool:
        """Use xprop to set the window type and state via the X11 window id."""
        surface = self.get_surface()
        if surface is None:
            return False

        try:
            from gi.repository import GdkX11  # noqa: F401 -- ensure X11 backend

            if not isinstance(surface, GdkX11.X11Surface):
                logger.warning("Not running on X11; skipping xprop calls.")
                return False

            xid = surface.get_xid()
        except (ImportError, AttributeError):
            logger.warning("GdkX11 not available; skipping xprop calls.")
            return False

        try:
            subprocess.Popen(
                [
                    "xprop",
                    "-id", str(xid),
                    "-f", "_NET_WM_WINDOW_TYPE", "32a",
                    "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_UTILITY",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.Popen(
                [
                    "xprop",
                    "-id", str(xid),
                    "-f", "_NET_WM_STATE", "32a",
                    "-set", "_NET_WM_STATE",
                    "_NET_WM_STATE_ABOVE,_NET_WM_STATE_STICKY,_NET_WM_STATE_SKIP_TASKBAR",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning("xprop not found; indicator may not stay above other windows.")

        return False  # do not repeat

    def _position_on_screen(self) -> bool:
        """Move the window to the saved position or the bottom-center default."""
        display = Gdk.Display.get_default()
        if display is None:
            return False

        monitors = display.get_monitors()
        if monitors.get_n_items() == 0:
            return False

        monitor = monitors.get_item(0)
        geo = monitor.get_geometry()

        if self._saved_position is not None:
            x = int(self._saved_position["x"])
            y = int(self._saved_position["y"])
        else:
            x = geo.x + (geo.width - _PILL_WIDTH) // 2
            y = geo.y + geo.height - _BOTTOM_MARGIN - _PILL_HEIGHT

        x, y = self._clamp_to_display(x, y)
        self._move_to(x, y)
        return False  # do not repeat

    def _move_to(self, x: int, y: int) -> None:
        """Move the indicator to absolute screen coordinates."""
        self._indicator_geo = (x, y, _PILL_WIDTH, _PILL_HEIGHT)

        surface = self.get_surface()
        moved = False
        if surface is not None:
            try:
                from gi.repository import GdkX11

                if isinstance(surface, GdkX11.X11Surface):
                    xid = surface.get_xid()
                    surface.move(x, y)
                    _move_x11_window(xid, x, y)
                    moved = True
            except (ImportError, AttributeError):
                pass
        if not moved:
            _move_indicator_window_by_name(x, y)

    def _clamp_to_display(self, x: int, y: int) -> tuple[int, int]:
        virtual_geo = _get_virtual_screen_geometry()
        if virtual_geo is not None:
            geo_x, geo_y, geo_width, geo_height = virtual_geo
            clamped_x = max(geo_x, min(x, geo_x + geo_width - _PILL_WIDTH))
            clamped_y = max(geo_y, min(y, geo_y + geo_height - _PILL_HEIGHT))
            return clamped_x, clamped_y

        display = Gdk.Display.get_default()
        if display is None:
            return x, y

        monitors = display.get_monitors()
        if monitors.get_n_items() == 0:
            return x, y

        min_x: Optional[int] = None
        min_y: Optional[int] = None
        max_x: Optional[int] = None
        max_y: Optional[int] = None
        for index in range(monitors.get_n_items()):
            monitor = monitors.get_item(index)
            geo = monitor.get_geometry()
            min_x = geo.x if min_x is None else min(min_x, geo.x)
            min_y = geo.y if min_y is None else min(min_y, geo.y)
            max_x = geo.x + geo.width if max_x is None else max(max_x, geo.x + geo.width)
            max_y = geo.y + geo.height if max_y is None else max(max_y, geo.y + geo.height)

        if min_x is None or min_y is None or max_x is None or max_y is None:
            return x, y

        clamped_x = max(min_x, min(x, max_x - _PILL_WIDTH))
        clamped_y = max(min_y, min(y, max_y - _PILL_HEIGHT))
        return clamped_x, clamped_y

    def _load_saved_position(self) -> Optional[dict[str, int]]:
        config = self._dbus_client.get_config()
        if not isinstance(config, dict):
            return None
        indicator = config.get("indicator")
        if not isinstance(indicator, dict):
            return None
        position = indicator.get("position")
        if not isinstance(position, dict):
            return None
        x = position.get("x")
        y = position.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        return {"x": x, "y": y}

    def _save_position(self) -> None:
        config = self._dbus_client.get_config()
        if not isinstance(config, dict):
            return

        x, y, _, _ = self._indicator_geo
        indicator = config.get("indicator")
        if not isinstance(indicator, dict):
            indicator = {}
            config["indicator"] = indicator
        indicator["position"] = {"x": int(x), "y": int(y)}

        def _on_saved(success: bool, result) -> None:
            if not success:
                logger.warning("Failed to persist indicator position: %s", result)

        self._dbus_client.save_config(config, callback=_on_saved)

    def _watch_pointer_drag(self) -> bool:
        pointer = _get_pointer_state()
        if pointer is None:
            return True

        pointer_x, pointer_y, window_id, button_down = pointer
        if self._drag_origin is None:
            if button_down and window_id == self.get_xid():
                x, y, _, _ = self._indicator_geo
                self._drag_origin = (x, y)
                self._drag_pointer_origin = (pointer_x, pointer_y)
            return True

        if button_down and self._drag_pointer_origin is not None:
            origin_x, origin_y = self._drag_origin
            pointer_origin_x, pointer_origin_y = self._drag_pointer_origin
            x, y = self._clamp_to_display(
                origin_x + pointer_x - pointer_origin_x,
                origin_y + pointer_y - pointer_origin_y,
            )
            if (x, y) != self._indicator_geo[:2]:
                self._saved_position = {"x": x, "y": y}
                self._move_to(x, y)
            return True

        self._drag_origin = None
        self._drag_pointer_origin = None
        self._save_position()
        return True

    # ------------------------------------------------------------------
    # Hover logic
    # ------------------------------------------------------------------

    def _on_mouse_enter(self, controller, x, y) -> None:
        if self._leave_timeout_id is not None:
            GLib.source_remove(self._leave_timeout_id)
            self._leave_timeout_id = None

    def _on_mouse_leave(self, controller) -> None:
        self._hide_hover_widgets()

    def _hide_hover_widgets(self) -> bool:
        """Hide separator and history button after mouse leaves."""
        # If the history panel is open, keep them visible.
        if self._history_panel is not None and self._history_panel.get_visible():
            return False
        self._separator.set_visible(False)
        self._history_btn.set_visible(False)
        self._leave_timeout_id = None
        return False  # do not repeat

    # ------------------------------------------------------------------
    # History panel
    # ------------------------------------------------------------------

    def get_xid(self) -> int:
        """Return the X11 window ID of this indicator window (0 if unavailable)."""
        surface = self.get_surface()
        if surface is None:
            return 0
        try:
            from gi.repository import GdkX11
            if isinstance(surface, GdkX11.X11Surface):
                return surface.get_xid()
        except (ImportError, AttributeError):
            pass
        return 0

    def _on_history_clicked(self, button: Gtk.Button) -> None:
        from bytecli.indicator.history_panel import HistoryPanel

        if self._history_panel is not None:
            self._history_panel.set_visible(not self._history_panel.get_visible())
            if self._history_panel.get_visible():
                self._history_panel.refresh()
            return

        self._history_panel = HistoryPanel(
            parent_window=self,
            dbus_client=self._dbus_client,
        )
        self._history_panel.connect("hide", self._on_history_hidden)
        self._history_panel.present()

    def _on_history_hidden(self, widget) -> None:
        # Trigger the leave-hide logic after the panel closes.
        self._leave_timeout_id = GLib.timeout_add(200, self._hide_hover_widgets)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state_idle(self) -> None:
        self._recording = False
        self._downloading = False
        self._transcribing = False
        self._stop_timer()
        self._stop_pulse()
        self._stop_wave()
        self._audio_level = 0.0
        self._clear_state_classes()
        self._status_label.set_text(i18n.t("indicator.idle", fallback="Idle"))
        self._timer_label.set_opacity(0.0)
        self._wave.set_opacity(0.0)
        self._dot.queue_draw()
        self._wave.queue_draw()
        self._queue_reposition()

    def set_state_downloading(self, percent: int, message: str) -> None:
        """Show model download progress in the pill."""
        self._recording = False
        self._downloading = True
        self._transcribing = False
        self._stop_timer()
        self._stop_pulse()
        self._stop_wave()
        self._audio_level = 0.0
        self._clear_state_classes()
        self._pill_box.add_css_class("indicator-pill-downloading")

        if percent < 0:
            # Error state.
            self._status_label.set_text("Download failed")
        elif percent < 100:
            self._status_label.set_text(f"Downloading... {percent}%")
        else:
            self._status_label.set_text("Loading model...")

        self._timer_label.set_opacity(0.0)
        self._wave.set_opacity(0.0)
        self._dot.queue_draw()
        self._wave.queue_draw()
        self._queue_reposition()

    def set_state_recording(self) -> None:
        self._recording = True
        self._downloading = False
        self._transcribing = False
        self._stop_pulse()
        self._clear_state_classes()
        self._pill_box.add_css_class("indicator-pill-recording")
        self._audio_level = 0.0
        self._wave_phase = 0.0
        self._timer_seconds = 0
        self._timer_label.set_text("00:00")
        self._timer_label.set_opacity(1.0)
        self._wave.set_opacity(1.0)
        self._status_label.set_text(i18n.t("indicator.recording", fallback="Recording"))
        self._dot.queue_draw()
        self._wave.queue_draw()
        self._queue_reposition()
        self._start_timer()
        self._start_wave()

    def set_state_transcribing(self) -> None:
        """Show the post-recording ASR inference state."""
        self._recording = False
        self._downloading = False
        self._transcribing = True
        self._stop_timer()
        self._stop_wave()
        self._clear_state_classes()
        self._pill_box.add_css_class("indicator-pill-transcribing")
        self._timer_label.set_opacity(0.0)
        self._wave.set_opacity(0.0)
        self._status_label.set_text(
            i18n.t("indicator.transcribing", fallback="Transcribing...")
        )
        self._start_pulse()
        self._dot.queue_draw()
        self._queue_reposition()

    def update_audio_level(self, level: float) -> None:
        if not self._recording:
            return
        clamped = max(0.0, min(1.0, float(level)))
        self._audio_level = (self._audio_level * 0.58) + (clamped * 0.42)
        self._wave.queue_draw()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer_source_id = GLib.timeout_add(1000, self._tick)

    def _stop_timer(self) -> None:
        if self._timer_source_id is not None:
            GLib.source_remove(self._timer_source_id)
            self._timer_source_id = None

    def _tick(self) -> bool:
        self._timer_seconds += 1
        mins, secs = divmod(self._timer_seconds, 60)
        self._timer_label.set_text(f"{mins:02d}:{secs:02d}")
        return True  # keep ticking

    def _start_pulse(self) -> None:
        self._stop_pulse()
        self._pulse_on = True
        self._pulse_source_id = GLib.timeout_add(520, self._pulse)

    def _stop_pulse(self) -> None:
        if self._pulse_source_id is not None:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None
        self._pulse_on = False

    def _start_wave(self) -> None:
        self._stop_wave()
        self._wave_source_id = GLib.timeout_add(42, self._animate_wave)

    def _stop_wave(self) -> None:
        if self._wave_source_id is not None:
            GLib.source_remove(self._wave_source_id)
            self._wave_source_id = None

    def _animate_wave(self) -> bool:
        if not self._recording:
            self._wave_source_id = None
            return False
        self._wave_phase += 0.34 + (self._audio_level * 0.28)
        self._audio_level *= 0.94
        self._wave.queue_draw()
        return True

    def _pulse(self) -> bool:
        if not self._transcribing:
            self._pulse_source_id = None
            return False
        self._pulse_on = not self._pulse_on
        self._dot.queue_draw()
        return True

    def _clear_state_classes(self) -> None:
        for css_class in (
            "indicator-pill-recording",
            "indicator-pill-transcribing",
            "indicator-pill-downloading",
        ):
            self._pill_box.remove_css_class(css_class)

    def _queue_reposition(self) -> None:
        GLib.idle_add(self._position_on_screen)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _apply_font_size(widget: Gtk.Widget, size_px: int) -> None:
    """Apply a pixel font size via an inline CSS provider."""
    provider = Gtk.CssProvider()
    css = f"* {{ font-size: {size_px}px; }}"
    provider.load_from_data(css.encode())
    widget.get_style_context().add_provider(
        provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )


def _rounded_rect(cr, x: float, y: float, width: float, height: float, radius: float) -> None:
    radius = min(radius, width / 2.0, height / 2.0)
    cr.new_sub_path()
    cr.arc(x + width - radius, y + radius, radius, -math.pi / 2.0, 0)
    cr.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2.0)
    cr.arc(x + radius, y + height - radius, radius, math.pi / 2.0, math.pi)
    cr.arc(x + radius, y + radius, radius, math.pi, 3.0 * math.pi / 2.0)
    cr.close_path()


def _move_x11_window(xid: int, x: int, y: int) -> None:
    try:
        subprocess.Popen(
            ["xdotool", "windowmove", str(xid), str(x), str(y)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.debug("xdotool not found; relying on GDK window move.")


def _move_indicator_window_by_name(x: int, y: int) -> None:
    try:
        subprocess.Popen(
            [
                "xdotool",
                "search",
                "--name",
                "ByteCLI Indicator",
                "windowmove",
                "%@",
                str(x),
                str(y),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.debug("xdotool not found; cannot move indicator by title.")


def _get_pointer_state() -> Optional[tuple[int, int, int, bool]]:
    global _X11_DISPLAY

    try:
        from Xlib import X, display

        if _X11_DISPLAY is None:
            _X11_DISPLAY = display.Display()

        pointer = _X11_DISPLAY.screen().root.query_pointer()
        window_id = int(pointer.child.id) if pointer.child else 0
        button_down = bool(pointer.mask & X.Button1Mask)
        return int(pointer.root_x), int(pointer.root_y), window_id, button_down
    except Exception as exc:
        logger.debug("Could not read X11 pointer state: %s", exc)
        _X11_DISPLAY = None
        return None


def _get_virtual_screen_geometry() -> Optional[tuple[int, int, int, int]]:
    global _SCREEN_GEOMETRY
    if _SCREEN_GEOMETRY is not None:
        return _SCREEN_GEOMETRY

    try:
        output = subprocess.check_output(
            ["xrandr"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    for line in output.splitlines():
        if " current " not in line:
            continue
        parts = line.split()
        try:
            current_index = parts.index("current")
            width = int(parts[current_index + 1])
            height = int(parts[current_index + 3].rstrip(","))
        except (ValueError, IndexError):
            return None
        _SCREEN_GEOMETRY = (0, 0, width, height)
        return _SCREEN_GEOMETRY

    return None
