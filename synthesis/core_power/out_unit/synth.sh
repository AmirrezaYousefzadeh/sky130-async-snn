#!/usr/bin/env bash
# LibreLane 3.0.16 synthesis only.
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh pdk librelane
TAG="${1:-synth}"
grep -q 8afc8346a57fe1ab7934ba5a6056ea8b43078e71 "$PDK_ROOT/sky130A/SOURCES" || { echo "PDK mismatch"; exit 1; }
nix --extra-experimental-features "nix-command flakes" develop --accept-flake-config "$LIBRELANE" -c \
  python3 -m librelane --pdk-root "$PDK_ROOT" --pdk sky130A --manual-pdk --run-tag "$TAG" --overwrite \
  --to Yosys.Synthesis config.yaml
