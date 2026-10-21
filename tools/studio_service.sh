#!/bin/sh
# Keep the local Studio API available independently of a terminal window.
set -eu

TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_LABEL=com.groxaxo.uniforge-studio
TASK_DOMAIN="gui/$(id -u)"
TASK_SERVICE="$TASK_DOMAIN/$TASK_LABEL"
TASK_PLIST="$HOME/Library/LaunchAgents/$TASK_LABEL.plist"
TASK_LOG="$TASK_REPO/run_logs/studio-server.log"
TASK_ACTION=${1:-status}

if [ "$(uname -s)" != Darwin ]; then
  echo 'This service launcher requires macOS. Use tools/start_studio.sh on other hosts.' >&2
  exit 1
fi

case "$TASK_ACTION" in
  install|start)
    if [ ! -x "$TASK_REPO/studio/.venv/bin/python" ] || [ ! -f "$TASK_REPO/studio/web/dist/index.html" ]; then
      echo 'Studio needs its local environment and frontend build. See studio/README.md.' >&2
      exit 1
    fi
    if ! launchctl print "$TASK_SERVICE" >/dev/null 2>&1 &&
       /usr/sbin/lsof -nP -iTCP:7861 -sTCP:LISTEN >/dev/null 2>&1; then
      echo 'Port 7861 is already in use. Stop the foreground Studio server before installing the service.' >&2
      exit 1
    fi
    mkdir -p "$(dirname "$TASK_PLIST")" "$(dirname "$TASK_LOG")"
    "$TASK_REPO/studio/.venv/bin/python" - "$TASK_REPO" "$TASK_PLIST" "$TASK_LOG" "$TASK_LABEL" <<'PY'
import os
from pathlib import Path
import plistlib
import sys

repo, filename, log, label = sys.argv[1:]
path = Path(filename)
config = {
    'Label': label,
    'ProgramArguments': ['/bin/sh', str(Path(repo) / 'tools/start_studio.sh')],
    'WorkingDirectory': repo,
    'RunAtLoad': True,
    'KeepAlive': True,
    'ThrottleInterval': 10,
    'StandardOutPath': log,
    'StandardErrorPath': log,
    'EnvironmentVariables': {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'},
}
if path.exists():
    current = plistlib.loads(path.read_bytes())
    if current != config:
        raise SystemExit(f'{path} already has a different configuration; review it before replacing it.')
else:
    with path.open('xb') as stream:
        plistlib.dump(config, stream)
    os.chmod(path, 0o600)
PY
    launchctl enable "$TASK_SERVICE"
    if ! launchctl print "$TASK_SERVICE" >/dev/null 2>&1; then
      launchctl bootstrap "$TASK_DOMAIN" "$TASK_PLIST"
    fi
    launchctl kickstart "$TASK_SERVICE"
    TASK_ATTEMPT=0
    until /usr/bin/curl --fail --silent --max-time 2 http://127.0.0.1:7861/api/jobs >/dev/null; do
      TASK_ATTEMPT=$((TASK_ATTEMPT + 1))
      if [ "$TASK_ATTEMPT" -ge 20 ]; then
        echo "Studio did not start. Check $TASK_LOG" >&2
        exit 1
      fi
      sleep 1
    done
    echo 'UniForge Studio is running: http://127.0.0.1:7861'
    echo 'It starts when you log in and restarts if the server exits.'
    ;;
  status)
    launchctl print "$TASK_SERVICE" | awk '/state =|pid =|last exit code =/'
    /usr/bin/curl --fail --silent --max-time 20 http://127.0.0.1:7861/api/health
    printf '\n'
    ;;
  stop)
    launchctl bootout "$TASK_SERVICE"
    echo 'Studio stopped for this login session. Use start to run it again.'
    ;;
  uninstall)
    if launchctl print "$TASK_SERVICE" >/dev/null 2>&1; then
      launchctl bootout "$TASK_SERVICE"
    fi
    rm -f "$TASK_PLIST"
    echo 'Studio login service removed. Project files and saved assets are retained.'
    ;;
  *)
    echo 'Usage: tools/studio_service.sh {install|start|status|stop|uninstall}' >&2
    exit 2
    ;;
esac
