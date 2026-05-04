#!/usr/bin/env bash
set -e

if [[ "${REQUIRE_XILINX:-0}" == "1" ]]; then
  SETTINGS_FILE="${XILINX_INSTALL:-}/Vitis/2022.1/settings64.sh"
  if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "Xilinx settings file not found: $SETTINGS_FILE" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$SETTINGS_FILE"

  if command -v vivado >/dev/null 2>&1; then
    vivado -version
  elif command -v vitis_hls >/dev/null 2>&1; then
    vitis_hls -version
  else
    echo "Xilinx tools are not available after sourcing $SETTINGS_FILE" >&2
    exit 1
  fi
fi

echo "Executing command: $@"
exec "$@"
