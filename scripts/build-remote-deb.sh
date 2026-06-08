#!/bin/bash
#
# ByteCLI local ASR variant .deb builder.
#
# This package is intentionally parallel-installable with the regular
# "bytecli" package:
#   - Debian package name: bytecli-remote-asr
#   - Python code path: /opt/bytecli-remote/lib/python3/dist-packages
#   - CLI wrappers: /usr/bin/bytecli-remote-*
#   - systemd user unit: bytecli-remote.service
#   - config/data dirs: ~/.config/bytecli-remote and ~/.local/share/bytecli-remote
#
# The D-Bus service name is still com.bytecli.Service, so do not run the
# regular and remote services at the same time.
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION="1.1.5-local6"
PY_PROJECT_VERSION="1.1.0"
PY_PACKAGE_NAME="bytecli"
PACKAGE_NAME="bytecli-remote-asr"
DEB_NAME="${PACKAGE_NAME}_${VERSION}_amd64.deb"
STAGING="${PROJECT_DIR}/staging-remote"
OPT_LIB="/opt/bytecli-remote/lib/python3/dist-packages"

info "Cleaning previous remote staging directory..."
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "${WHEEL_DIR}"' EXIT

info "Building Python wheel for remote staging..."
/usr/bin/python3 -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "${WHEEL_DIR}" \
    "${PROJECT_DIR}" 2>&1 | grep -v '^\[notice\]' || true

WHEEL_FILE=$(find "${WHEEL_DIR}" -maxdepth 1 -name "${PY_PACKAGE_NAME}-${PY_PROJECT_VERSION}-*.whl" | head -1)
if [ -z "${WHEEL_FILE}" ]; then
    error "Could not build Python wheel"
fi

DIST_PACKAGES="${STAGING}${OPT_LIB}"
mkdir -p "${DIST_PACKAGES}"

/usr/bin/python3 - "${WHEEL_FILE}" "${DIST_PACKAGES}" << 'PY'
import sys
import zipfile

wheel_file, target_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel_file) as zf:
    zf.extractall(target_dir)
PY
success "Python package installed to ${DIST_PACKAGES}"

info "Creating remote wrapper scripts..."
mkdir -p "${STAGING}/usr/bin"

make_wrapper() {
    local path="$1"
    local import_target="$2"
    cat > "${path}" << WRAPPER
#!/usr/bin/python3
import os
import sys

remote_path = "${OPT_LIB}"
os.environ.setdefault("BYTECLI_CONFIG_DIR", os.path.expanduser("~/.config/bytecli-remote"))
os.environ.setdefault("BYTECLI_DATA_DIR", os.path.expanduser("~/.local/share/bytecli-remote"))
os.environ.setdefault("BYTECLI_PROFILE_SET", "remote")
os.environ.setdefault("BYTECLI_RUNTIME_NAME", "bytecli-remote")
existing = os.environ.get("PYTHONPATH", "")
if remote_path not in existing.split(":"):
    os.environ["PYTHONPATH"] = remote_path + ((":" + existing) if existing else "")
sys.path.insert(0, remote_path)

from ${import_target} import main
raise SystemExit(main())
WRAPPER
    chmod 755 "${path}"
}

make_wrapper "${STAGING}/usr/bin/bytecli-remote-service" "bytecli.service.main"
make_wrapper "${STAGING}/usr/bin/bytecli-remote-indicator" "bytecli.indicator.main"
make_wrapper "${STAGING}/usr/bin/bytecli-remote-settings" "bytecli.settings.main"
make_wrapper "${STAGING}/usr/bin/bytecli-remote-asr-eval" "bytecli.eval.asr_eval"
install -m 755 "${PROJECT_DIR}/scripts/download-sherpa-onnx-models.sh" \
    "${STAGING}/usr/bin/bytecli-remote-download-sherpa-models"
success "Remote wrapper scripts created"

info "Installing local ASR systemd user service..."
mkdir -p "${STAGING}/lib/systemd/user"
cat > "${STAGING}/lib/systemd/user/bytecli-remote.service" << SERVICE
[Unit]
Description=ByteCLI Local Qwen/Fun/Sherpa ASR Voice Dictation Service
After=graphical-session.target
StartLimitBurst=3
StartLimitIntervalSec=60

[Service]
Type=simple
ExecStart=/usr/bin/bytecli-remote-service
Environment=DISPLAY=:1
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus
Environment=BYTECLI_CONFIG_DIR=%h/.config/bytecli-remote
Environment=BYTECLI_DATA_DIR=%h/.local/share/bytecli-remote
Environment=BYTECLI_PROFILE_SET=remote
Environment=BYTECLI_RUNTIME_NAME=bytecli-remote
Environment=PYTHONPATH=${OPT_LIB}
Environment=LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICE
success "Local ASR systemd service file installed"

info "Installing remote desktop entry..."
mkdir -p "${STAGING}/usr/share/applications"
cat > "${STAGING}/usr/share/applications/bytecli-remote-settings.desktop" << 'DESKTOP'
[Desktop Entry]
Type=Application
Name=ByteCLI Local ASR Settings
Comment=Configure ByteCLI local Qwen/Fun/Sherpa ASR dictation
Exec=/usr/bin/bytecli-remote-settings
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;
DESKTOP

info "Generating remote DEBIAN control files..."
mkdir -p "${STAGING}/DEBIAN"
cat > "${STAGING}/DEBIAN/control" << CONTROL
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.10), python3-gi, gir1.2-gtk-4.0, gir1.2-adw-1,
         python3-dbus, xclip, xdotool, x11-utils, libportaudio2, python3-numpy,
         python3-pip
Description: ByteCLI local Qwen/Fun/Sherpa ASR dictation variant
 Parallel-installable ByteCLI variant exposing local Qwen3-ASR-0.6B,
 Fun-ASR-Nano, and sherpa-onnx ASR profiles. It installs separate command
 names, systemd unit, and config/data directories from the regular bytecli
 package.
Maintainer: ByteCLI <noreply@github.com>
Homepage: https://github.com/StriderXOXO/byteCLI
CONTROL

cat > "${STAGING}/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
set -e
case "$1" in
  configure)
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --global daemon-reload 2>/dev/null || true
    echo "ByteCLI local Qwen/Fun ASR variant installed."
    echo "It is not auto-started. Stop bytecli.service, then start bytecli-remote.service."
    ;;
esac
exit 0
POSTINST

cat > "${STAGING}/DEBIAN/prerm" << 'PRERM'
#!/bin/bash
set -e
case "$1" in
  remove|purge)
    systemctl --global disable bytecli-remote.service 2>/dev/null || true
    if [ -n "$SUDO_USER" ]; then
        SUDO_UID=$(id -u "$SUDO_USER")
        if [ -d "/run/user/$SUDO_UID" ]; then
            su - "$SUDO_USER" -c "systemctl --user stop bytecli-remote.service" 2>/dev/null || true
        fi
    fi
    ;;
esac
exit 0
PRERM

chmod 755 "${STAGING}/DEBIAN/postinst" "${STAGING}/DEBIAN/prerm"
success "Remote DEBIAN control files ready"

info "Fixing file permissions..."
find "${STAGING}" -type d -exec chmod 755 {} \;
find "${STAGING}/opt" -type f -exec chmod 644 {} \; 2>/dev/null || true
find "${STAGING}/usr/share" -type f -exec chmod 644 {} \; 2>/dev/null || true
find "${STAGING}/lib" -type f -exec chmod 644 {} \; 2>/dev/null || true
success "Permissions fixed"

info "Building remote .deb package..."
dpkg-deb --build "${STAGING}" "${PROJECT_DIR}/${DEB_NAME}"

echo ""
echo -e "${GREEN}${BOLD}========================================${NC}"
echo -e "${GREEN}${BOLD}  Remote .deb package built successfully!${NC}"
echo -e "${GREEN}${BOLD}========================================${NC}"
echo ""
echo -e "Output: ${BOLD}${PROJECT_DIR}/${DEB_NAME}${NC}"
echo -e "Install: ${BOLD}sudo dpkg -i ${DEB_NAME}${NC}"
echo ""

rm -rf "${STAGING}"
success "Remote staging directory cleaned up"
