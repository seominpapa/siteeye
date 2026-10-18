#!/usr/bin/env bash
# Double-click this file in Finder (macOS) or run ./siteeye.command: starts the web app and opens it in your browser.
# Closing the window or pressing Ctrl+C stops the server.
#
# You may copy this file anywhere (for example the Desktop), or make a symlink / Finder alias to it. A copy finds the
# project through the path remembered the last time the launcher ran inside the project folder
# (~/.config/siteeye/home), or through SITEEYE_HOME.
#   PORT=8010 ./siteeye.command      use another starting port (default 8000)
#   NO_BROWSER=1 ./siteeye.command   do not open a browser (headless / testing)
#   SITEEYE_HOME=/path/to/siteeye    point a copied launcher at the project explicitly

pause_and_exit() { printf '\n%s\n' "$1"; read -r -p "Enter 를 누르면 창을 닫습니다 " _; exit 1; }

# 0) Locate the project folder
SELF="$0"
while [ -L "$SELF" ]; do                                    # follow symlinks to the real file
  LINK="$(readlink "$SELF")"
  case "$LINK" in /*) SELF="$LINK" ;; *) SELF="$(dirname "$SELF")/$LINK" ;; esac
done
HERE="$(cd "$(dirname "$SELF")" 2>/dev/null && pwd)"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/siteeye/home"

if [ -f "$HERE/main.py" ] && [ -d "$HERE/siteeye" ]; then
  PROJECT="$HERE"
  mkdir -p "$(dirname "$CONF")" 2>/dev/null && printf '%s\n' "$PROJECT" > "$CONF"   # remember it for copies of this file
elif [ -n "$SITEEYE_HOME" ] && [ -f "$SITEEYE_HOME/main.py" ]; then
  PROJECT="$SITEEYE_HOME"
elif [ -f "$CONF" ] && [ -f "$(cat "$CONF")/main.py" ]; then
  PROJECT="$(cat "$CONF")"
else
  pause_and_exit "siteeye 프로젝트 폴더를 찾지 못했습니다.
이 파일을 프로젝트 폴더 안에서 한 번 실행하면 위치가 기억되어 복사본도 동작합니다.
또는 SITEEYE_HOME=/프로젝트/경로 ./siteeye.command 로 지정하세요."
fi
cd "$PROJECT" || exit 1

HOST=127.0.0.1
START_PORT="${PORT:-8000}"

# 1) Python environment (first run only)
if [ ! -x .venv/bin/python ]; then
  echo "처음 실행입니다. 가상환경을 만들고 패키지를 설치합니다 (몇 분 걸릴 수 있습니다)..."
  PY=""
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || pause_and_exit "Python 3.10 이상이 필요합니다. https://www.python.org 에서 설치한 뒤 다시 실행하세요."
  "$PY" -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt \
    || pause_and_exit "패키지 설치에 실패했습니다. 인터넷 연결과 git 설치를 확인하세요."
fi

# 2) Pick a port: reuse our own running server, otherwise the first free port
is_ours() { curl -fs --max-time 1 "http://$HOST:$1/healthz" 2>/dev/null | grep -q '"ok"'; }
is_busy() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('$HOST', $1)) == 0 else 1)"; }
open_url() {
  [ -n "$NO_BROWSER" ] && return 0
  if command -v open >/dev/null 2>&1; then open "$1"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1; else echo "브라우저에서 $1 을 여세요."; fi
}

PORT_FOUND=""
for p in $(seq "$START_PORT" $((START_PORT + 10))); do
  if is_ours "$p"; then
    echo "이미 실행 중인 서버가 있습니다: http://$HOST:$p"
    open_url "http://$HOST:$p"
    exit 0
  fi
  if ! is_busy "$p"; then PORT_FOUND="$p"; break; fi
done
[ -n "$PORT_FOUND" ] || pause_and_exit "$START_PORT~$((START_PORT + 10)) 포트가 모두 사용 중입니다."

# 3) Start the server, wait until it answers, then open the browser
.venv/bin/python main.py --web --port "$PORT_FOUND" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 120); do
  is_ours "$PORT_FOUND" && break
  kill -0 "$SERVER_PID" 2>/dev/null || pause_and_exit "서버가 시작되지 않았습니다. 위 메시지를 확인하세요."
  sleep 1
done
is_ours "$PORT_FOUND" || pause_and_exit "서버가 2분 안에 응답하지 않았습니다."

URL="http://$HOST:$PORT_FOUND"
echo
echo "서버가 실행 중입니다: $URL"
echo "(이 창을 닫거나 Ctrl+C 를 누르면 서버가 종료됩니다)"
open_url "$URL"
wait "$SERVER_PID"
