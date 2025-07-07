#!/usr/bin/env bash
set -e

# 0) Ensure we have a bashrc to edit
BASHRC="${HOME}/.bashrc"
touch "$BASHRC"

# 1) Add Vitis settings source to .bashrc if not already present
SOURCE_LINE="source \"${XILINX_INSTALL}/Vitis/2022.1/settings64.sh\""
grep -qxF "$SOURCE_LINE" "$BASHRC" \
  || echo "$SOURCE_LINE" >> "$BASHRC"

# 2) Now source the bashrc (which in turn sources Vitis settings)
#    This makes vitis_hls et al. available in the environment.
# shellcheck disable=SC1090
. "$BASHRC"
# cat $BASHRC

# Check Vivado version
vivado -version

# 3) Exec the requested CMD (defaults to `bash -l` if none provided)
echo "Executing command: $@"
exec "$@"