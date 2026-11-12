# Tool and PDK locations of the measurement flow, sourced by every script here (`source ./env.sh [pdk] [librelane]`).
# Set them in the environment or in env.local.sh next to this file (not tracked):
#   PDK_ROOT       open_pdks install that contains sky130A (the measurements used open_pdks commit 8afc834)
#   LIBRELANE      LibreLane 3.0.16 source tree; its Nix flake provides OpenROAD and OpenSTA
#   OSS_CAD_SUITE  YosysHQ OSS CAD Suite (2026-10-24 was used: Yosys, Icarus Verilog); unset = the tools on PATH
_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_env_dir/env.local.sh" ]]; then source "$_env_dir/env.local.sh"; fi
if [[ -n "${OSS_CAD_SUITE:-}" ]]; then export PATH="$OSS_CAD_SUITE/bin:$PATH"; fi
PDK_ROOT="${PDK_ROOT:-}"; LIBRELANE="${LIBRELANE:-}"
LIB="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
PDKV="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/verilog"
for _need in "$@"; do
  case "$_need" in
    pdk) [[ -d "$PDK_ROOT/sky130A" ]] || { echo "env.sh: set PDK_ROOT to the open_pdks root that contains sky130A" >&2; exit 1; } ;;
    librelane) [[ -f "$LIBRELANE/flake.nix" ]] || { echo "env.sh: set LIBRELANE to a LibreLane 3.0.16 source tree" >&2; exit 1; }
      if [[ -f /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]]; then
        source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh; fi ;;
  esac
done
