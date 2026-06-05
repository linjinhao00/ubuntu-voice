#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DATA_DIR="${HOME}/.local/share/bytecli"
if [[ "$(basename "$0")" == "bytecli-remote-download-sherpa-models" ]]; then
    DEFAULT_DATA_DIR="${HOME}/.local/share/bytecli-remote"
fi

DATA_DIR="${BYTECLI_DATA_DIR:-${DEFAULT_DATA_DIR}}"
MODEL_DIR="${DATA_DIR}/models"
DOWNLOAD_DIR="${BYTECLI_DOWNLOAD_DIR:-/tmp/bytecli-sherpa-downloads}"

SENSEVOICE_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
FUNASR_NAME="sherpa-onnx-funasr-nano-int8-2025-12-30"

SENSEVOICE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SENSEVOICE_NAME}.tar.bz2"
FUNASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${FUNASR_NAME}.tar.bz2"

download_and_extract() {
    local name="$1"
    local url="$2"
    local archive="${DOWNLOAD_DIR}/${name}.tar.bz2"
    local target="${MODEL_DIR}/${name}"

    mkdir -p "${MODEL_DIR}" "${DOWNLOAD_DIR}"
    if [[ -d "${target}" ]]; then
        echo "${name} already exists at ${target}"
        return
    fi

    echo "Downloading ${name}..."
    curl -L --fail --continue-at - -o "${archive}" "${url}"

    echo "Extracting ${name}..."
    tar -xjf "${archive}" -C "${MODEL_DIR}"
}

download_and_extract "${SENSEVOICE_NAME}" "${SENSEVOICE_URL}"
download_and_extract "${FUNASR_NAME}" "${FUNASR_URL}"

echo "Done. Models are under ${MODEL_DIR}"
