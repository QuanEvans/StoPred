#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/esmc_600m_2024_12_v0.pth"
MODEL_REPO="${ESMC_HF_REPO:-biohub/esmc-600m-2024-12}"
MODEL_FILE="${ESMC_HF_FILE:-data/weights/esmc_600m_2024_12_v0.pth}"
REVISION="${ESMC_HF_REVISION:-main}"
OUTPUT_PATH="${DEFAULT_OUTPUT}"
FORCE=0

usage() {
    cat <<EOF
Usage: bash external/download_esmc_600m.sh [options]

Download the ESM-C 600M weights used by StoPred.

Options:
  -o, --output PATH   Output path. Default: ${DEFAULT_OUTPUT}
  -f, --force         Re-download even if the output file already exists.
  -h, --help          Show this help message.

Environment:
  HF_TOKEN            Optional Hugging Face token for authenticated downloads.
  ESMC_HF_REPO        Hugging Face repo. Default: ${MODEL_REPO}
  ESMC_HF_FILE        File inside the repo. Default: ${MODEL_FILE}
  ESMC_HF_REVISION    Repo revision. Default: ${REVISION}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -e "${OUTPUT_PATH}" && "${FORCE}" -ne 1 ]]; then
    echo "Found existing file: ${OUTPUT_PATH}"
    echo "Use --force to re-download."
    exit 0
fi

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
PART_PATH="${OUTPUT_PATH}.part"
URL="https://huggingface.co/${MODEL_REPO}/resolve/${REVISION}/${MODEL_FILE}"
mkdir -p "${OUTPUT_DIR}"

echo "Downloading ESM-C 600M weights"
echo "  source: ${URL}"
echo "  output: ${OUTPUT_PATH}"

if command -v curl >/dev/null 2>&1; then
    CURL_ARGS=(
        --location
        --fail
        --continue-at -
        --retry 5
        --retry-delay 5
        --output "${PART_PATH}"
    )
    if [[ -n "${HF_TOKEN:-}" ]]; then
        CURL_ARGS+=(--header "Authorization: Bearer ${HF_TOKEN}")
    fi
    curl "${CURL_ARGS[@]}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    WGET_ARGS=(
        --continue
        --output-document="${PART_PATH}"
    )
    if [[ -n "${HF_TOKEN:-}" ]]; then
        WGET_ARGS+=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi
    wget "${WGET_ARGS[@]}" "${URL}"
else
    echo "Neither curl nor wget is available." >&2
    exit 1
fi

mv "${PART_PATH}" "${OUTPUT_PATH}"
echo "Downloaded: ${OUTPUT_PATH}"
