#!/bin/sh
# Independent CPU-only Mac queue. GPU admission belongs to .51 GPUQ.
set -eu
TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_LABEL=com.groxaxo.uniforge-studio-3090
TASK_DOMAIN="gui/$(id -u)"
TASK_PLIST="$HOME/Library/LaunchAgents/$TASK_LABEL.plist"
TASK_ACTION=${1:-status}
case "$TASK_ACTION" in
  install|start)
    if ! launchctl print "$TASK_DOMAIN/$TASK_LABEL" >/dev/null 2>&1 &&
       /usr/sbin/lsof -nP -iTCP:7864 -sTCP:LISTEN >/dev/null 2>&1; then
      echo 'Port 7864 is already in use; review that listener before continuing.' >&2
      exit 1
    fi
    mkdir -p "$HOME/Library/LaunchAgents" "$TASK_REPO/run_logs"
    python3 - "$TASK_REPO" "$TASK_PLIST" "$TASK_LABEL" <<'PY'
from pathlib import Path
import os,plistlib,sys
repo,path,label=sys.argv[1:]
config=dict(Label=label, ProgramArguments=['/bin/sh',str(Path(repo)/'tools/start_studio_3090.sh')],
    WorkingDirectory=repo, RunAtLoad=True, KeepAlive=True, ThrottleInterval=10,
    StandardOutPath=str(Path(repo)/'run_logs/studio-3090.log'),
    StandardErrorPath=str(Path(repo)/'run_logs/studio-3090.log'),
    EnvironmentVariables={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin'})
p=Path(path)
if p.exists() and plistlib.loads(p.read_bytes())!=config:
    raise SystemExit('Existing launch service differs; review before replacing it')
if not p.exists():
    p.write_bytes(plistlib.dumps(config)); os.chmod(p,0o600)
PY
    launchctl enable "$TASK_DOMAIN/$TASK_LABEL"
    if ! launchctl print "$TASK_DOMAIN/$TASK_LABEL" >/dev/null 2>&1; then
      launchctl bootstrap "$TASK_DOMAIN" "$TASK_PLIST"
    fi
    launchctl kickstart "$TASK_DOMAIN/$TASK_LABEL"
    echo 'RTX 3090 GPUQ queue started: http://100.85.200.56:7864'
    ;;
  status)
    launchctl print "$TASK_DOMAIN/$TASK_LABEL" | awk '/state =|pid =|last exit code =/'
    curl --fail --silent --show-error --max-time 45 http://127.0.0.1:7864/api/health
    ;;
  stop) launchctl bootout "$TASK_DOMAIN/$TASK_LABEL" ;;
  *) echo 'Usage: studio_3090_service.sh {install|start|status|stop}' >&2; exit 2 ;;
esac
