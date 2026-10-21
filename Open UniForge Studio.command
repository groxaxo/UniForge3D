#!/bin/sh
set -eu
TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$TASK_REPO/tools/studio_service.sh" start
open http://127.0.0.1:7861
