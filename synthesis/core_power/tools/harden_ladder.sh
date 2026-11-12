#!/usr/bin/env bash
# Harden a design with LibreLane: utilisations tried from
# dense to sparse, every attempt watched by the routing watchdog and stopped by its process group, the first attempt
# that passes the acceptance checks accepted, every attempt logged. It runs for hours: start it as a batch job.
#
# Usage: harden_ladder.sh <config.yaml|json> <work dir> "<utilisations, dense to sparse, e.g. 50 40 30 20>"
# Result: <work dir>/runs/accepted -> the accepted run (absent if none passed), <work dir>/ladder.json (all attempts),
#         <work dir>/logs/librelane_u<U>.log per attempt. Exit status 0 if a run was accepted, 1 otherwise.
# Acceptance (from the run's final/metrics.json): route__drc_errors 0, timing__setup__ws >= 0 and timing__hold__ws >= 0
# (LibreLane's worst over all its STA corners, setup included: stricter than its own checkers, which check setup at
# the typical corners only and do not run with --to OpenROAD.STAPostPNR), design__power_grid_violation__count 0, and a
# STAPostPNR step in the run (without it, RUN_MCSTA: false or an earlier LAST_STEP, the slacks are pre-placement
# values; LibreLane issue #1040). If SIGNOFF_CMD is set, it is
# run as `$SIGNOFF_CMD <run dir>` after those checks and must exit 0 as well (your own sign-off).
# A crashed attempt (a signal such as "Signal 11 received", "Segmentation fault", "Illegal instruction", "stack
# smashing") says nothing about routability: the ladder stops there instead of trying a sparser utilisation. So does
# an attempt that LibreLane stops for disconnected pins after routing: in OpenROAD dcf3613 that is CTS losing a clock
# gate's clock input (OpenROAD issue #9816), a tool bug and not a routing result; PRE_CTS_TCL works around it.
# Existing runs with the ladder's tags (runs/u<U> beside the configuration) are never overwritten unless OVERWRITE=1.
# Environment (defaults in brackets):
#   LIBRELANE (required)   PDK_ROOT (required)   PDK [sky130A]
#   DENSITY_ADD [10]: PL_TARGET_DENSITY_PCT = utilisation + DENSITY_ADD, at most 95
#   LAST_STEP [OpenROAD.STAPostPNR]   EXTRA_OVERRIDES: more "KEY=VALUE" overrides, space separated (JSON values)
#   PRE_CTS_TCL: an OpenROAD script run on the placed database before CTS (ODB=<file>, rewritten in place); the attempt
#   then runs in two LibreLane calls on the same run directory (--to OpenROAD.DetailedPlacement, --from OpenROAD.CTS).
#   tools/firm_clock_gates.tcl is the workaround for the lost clock-gate inputs above.
#   watchdog: WD_VIOL3 [500000] after 3 routing iterations, WD_VIOL6 [20000] after 6, WD_VIOL10 [3000] after 10,
#             WD_IDLE_H [12] hours without a finished iteration, WD_MAX_H [36] hours for any attempt
#   HARDEN_THREADS [4]: routing threads (DRT_THREADS) and LibreLane's worker limit (-j; also caps the parallel
#   STA and RCX corners, whose STA_THREADS default would otherwise be every core)
set -euo pipefail
[[ $# -eq 3 ]] || { sed -n '6,7p' "$0"; exit 2; }
CFG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
mkdir -p "$2/logs"; WORK="$(cd "$2" && pwd)"; UTILS="$3"
LIBRELANE="${LIBRELANE:?set LIBRELANE to a LibreLane source tree}"
PDK_ROOT="${PDK_ROOT:?set PDK_ROOT to the open_pdks root that contains sky130A}"; export PDK="${PDK:-sky130A}"
LAST_STEP="${LAST_STEP:-OpenROAD.STAPostPNR}"
THREADS="${HARDEN_THREADS:-${JOBQ_THREADS:-4}}"
for U in $UTILS; do [[ "$U" =~ ^[0-9]+$ ]] || { echo "utilisations must be whole numbers: $UTILS"; exit 2; }; done
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh

ACTIVE=""
stop_group() {     # <pgid>: TERM to the whole process group, KILL after a minute
  kill -TERM -- "-$1" 2> /dev/null || return 0
  for _ in {1..12}; do kill -0 -- "-$1" 2> /dev/null || return 0; sleep 5; done
  kill -KILL -- "-$1" 2> /dev/null || true
}
trap '[[ -n "$ACTIVE" ]] && stop_group "$ACTIVE"' INT TERM EXIT

watchdog() {       # <run dir>: stop the active attempt when detailed routing cannot converge; prints the reason
  local run="$1" seen=0 last_change start now log reason prev gap
  last_change=$(date +%s); start=$last_change; prev=$last_change
  while kill -0 "$ACTIVE" 2> /dev/null; do
    sleep 60
    now=$(date +%s); reason=""
    # time spent frozen by a job scheduler (SIGSTOP; the watchdog is frozen too) is neither idle nor run time
    gap=$(( now - prev - 60 )); prev=$now
    (( gap > 180 )) && { last_change=$(( last_change + gap )); start=$(( start + gap )); }
    log=$(ls -t "$run"/*openroad-detailedrouting*/*.log 2> /dev/null | head -n 1 || true)
    if [[ -n "$log" ]]; then
      local v=() n
      mapfile -t v < <(grep "Number of violations" "$log" | awk '{print $NF}' | tr -d '.,')
      n=${#v[@]}
      [[ $n -ne $seen ]] && { seen=$n; last_change=$now; }
      (( n >= 3 && v[n-1] > ${WD_VIOL3:-500000} )) && reason="${v[n-1]} routing violations after $n iterations"
      (( n >= 6 && v[n-1] > ${WD_VIOL6:-20000} )) && reason="${v[n-1]} routing violations after $n iterations"
      (( n >= 10 && v[n-1] > ${WD_VIOL10:-3000} )) && reason="${v[n-1]} routing violations after $n iterations"
      (( now - last_change > ${WD_IDLE_H:-12} * 3600 )) && reason="no completed routing iteration for ${WD_IDLE_H:-12} h"
    else
      last_change=$now
    fi
    (( now - start > ${WD_MAX_H:-36} * 3600 )) && reason="longer than ${WD_MAX_H:-36} h"
    if [[ -n "$reason" ]]; then stop_group "$ACTIVE"; echo "$reason"; return; fi
  done
}

check() {          # <run dir>: prints "pass" or the reason for rejecting the run
  python3 - "$1/final/metrics.json" "$1" <<'PY'
import glob, json, os, sys
try:
    m = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("no final metrics (the flow stopped before sign-off)"); sys.exit()
problems = []
if not glob.glob(os.path.join(sys.argv[2], "*-openroad-stapostpnr*")):
    problems.append("no STAPostPNR step (RUN_MCSTA off or LAST_STEP earlier): slacks are pre-placement values")
def get(key):
    if key not in m:
        problems.append(f"{key} missing")
    return m.get(key)
drc, setup, hold, pdn = (get(k) for k in ("route__drc_errors", "timing__setup__ws", "timing__hold__ws",
                                          "design__power_grid_violation__count"))
if drc: problems.append(f"{drc} routing DRC errors")
if setup is not None and setup < 0: problems.append(f"setup slack {setup:.3f} ns")
if hold is not None and hold < 0: problems.append(f"hold slack {hold:.3f} ns")
if pdn: problems.append(f"{pdn} power-grid violations")
print("; ".join(problems) if problems else "pass")
PY
}

if [[ "${OVERWRITE:-0}" != 1 ]]; then
  for U in $UTILS; do
    [[ -e "$(dirname "$CFG")/runs/u$U" ]] && { echo "run $(dirname "$CFG")/runs/u$U exists; move it away or set OVERWRITE=1"; exit 2; }
  done
fi
ATTEMPTS="$WORK/ladder_attempts.tsv"; : > "$ATTEMPTS"
CRASHED=""
rm -f "$WORK/runs/accepted"
ACCEPTED=""
for U in $UTILS; do
  DENS=$(( U + ${DENSITY_ADD:-10} )); (( DENS > 95 )) && DENS=95
  TAG="u$U"; LOG="$WORK/logs/librelane_$TAG.log"; T0=$(date +%s)
  echo "==== utilisation $U %, density $DENS %, run $TAG  ($(date '+%F %H:%M'))"
  OVR=(--override-config "FP_CORE_UTIL=$U" --override-config "PL_TARGET_DENSITY_PCT=$DENS"
       --override-config "DRT_THREADS=$THREADS")
  for o in ${EXTRA_OVERRIDES:-}; do OVR+=(--override-config "$o"); done
  ( cd "$WORK" && export _OPENLANE_MAX_CORES="$THREADS" THREADS PDK_ROOT TAG LIBRELANE LAST_STEP \
      PRE_CTS_TCL="${PRE_CTS_TCL:-}" && exec setsid nice -n 19 bash -c '
    set -euo pipefail
    cfg=$1; shift
    dev=(nix --extra-experimental-features "nix-command flakes" develop --accept-flake-config "$LIBRELANE" -c)
    ll=(python3 -m librelane -j "$THREADS" --pdk-root "$PDK_ROOT" --pdk "$PDK" --manual-pdk --run-tag "$TAG")
    [[ -n "$PRE_CTS_TCL" ]] || exec "${dev[@]}" "${ll[@]}" --overwrite --to "$LAST_STEP" "$@" "$cfg"
    "${dev[@]}" "${ll[@]}" --overwrite --to OpenROAD.DetailedPlacement "$@" "$cfg"
    odb=$(ls -t "$(dirname "$cfg")/runs/$TAG"/*-openroad-detailedplacement/*.odb | head -n 1)
    echo "PRE_CTS_TCL: $PRE_CTS_TCL on $odb"
    ODB="$odb" "${dev[@]}" openroad -exit -no_splash "$PRE_CTS_TCL"
    exec "${dev[@]}" "${ll[@]}" --from OpenROAD.CTS --to "$LAST_STEP" "$@" "$cfg"' bash "$CFG" "${OVR[@]}" ) > "$LOG" 2>&1 &
  ACTIVE=$!
  STOPPED=$(watchdog "$(dirname "$CFG")/runs/$TAG")
  wait "$ACTIVE" || true; ACTIVE=""
  RUN="$(dirname "$CFG")/runs/$TAG"
  if [[ -n "$STOPPED" ]]; then
    VERDICT="stopped by the watchdog: $STOPPED"
  else
    VERDICT=$(check "$RUN")
    CRASH=$(grep -m1 -aoE 'Signal (4|6|7|8|11) received|Segmentation fault|Illegal instruction|stack smashing detected|Bus error' "$LOG" || true)
    if [[ "$VERDICT" != pass && -n "$CRASH" ]]; then
      VERDICT="crashed ($CRASH): not a routability result, ladder stopped; rerun this utilisation"; CRASHED=$U
    elif [[ "$VERDICT" != pass ]] && grep -qa 'critical disconnected pins found' "$LOG"; then
      PINS=$(grep -ahoE "instance '[^']+' are not connected" "$RUN/error.log" "$LOG" 2> /dev/null | sort -u | tr '\n' ' ' || true)
      VERDICT="disconnected pins after routing ($PINS): a tool bug, not a routability result (OpenROAD issue #9816), ladder stopped"
      CRASHED=$U
    fi
    if [[ "$VERDICT" == pass && -n "${SIGNOFF_CMD:-}" ]]; then
      $SIGNOFF_CMD "$RUN" > "$WORK/logs/signoff_$TAG.log" 2>&1 || VERDICT="own sign-off failed (logs/signoff_$TAG.log)"
    fi
  fi
  MIN=$(( ($(date +%s) - T0) / 60 ))
  printf '%s\t%s\t%s\t%s\n' "$U" "$DENS" "$MIN" "$VERDICT" >> "$ATTEMPTS"
  echo "   $VERDICT  ($MIN min)"
  if [[ "$VERDICT" == pass ]]; then
    mkdir -p "$WORK/runs"; ln -sfn "$RUN" "$WORK/runs/accepted"; ACCEPTED=$U; break
  fi
  [[ -n "$CRASHED" ]] && break
done

python3 - "$ATTEMPTS" "$WORK/ladder.json" "$CFG" "$PDK_ROOT" "$LIBRELANE" "${ACCEPTED:-}" <<'PY'
import json, os, subprocess, sys
attempts_file, out, cfg, pdk_root, librelane, accepted = sys.argv[1:]
attempts = [dict(zip(("utilisation_pct", "density_pct", "minutes", "verdict"), line.rstrip("\n").split("\t")))
            for line in open(attempts_file) if line.strip()]
for a in attempts:
    for k in ("utilisation_pct", "density_pct", "minutes"):
        a[k] = int(a[k])
sources = os.path.join(pdk_root, os.environ.get("PDK", "sky130A"), "SOURCES")
summary = {"config": cfg, "accepted_utilisation_pct": int(accepted) if accepted else None, "attempts": attempts,
           "librelane": os.path.realpath(librelane),
           "pdk": open(sources).read().strip() if os.path.exists(sources) else os.path.realpath(pdk_root)}
json.dump(summary, open(out, "w"), indent=1)
PY
rm -f "$ATTEMPTS"
[[ -n "$ACCEPTED" ]] && { echo "accepted at $ACCEPTED %: $WORK/runs/accepted"; exit 0; }
[[ -n "$CRASHED" ]] && { echo "stopped at $CRASHED % by a crash or a tool bug (see $WORK/logs/librelane_u$CRASHED.log); see $WORK/ladder.json"; exit 1; }
echo "no utilisation passed; see $WORK/ladder.json"; exit 1
