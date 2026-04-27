#!/bin/bash
# Real-time monitor for Waveline STT server
# Usage: ./monitor.sh [logfile]

LOG="${1:-transcriptions.log}"

if [[ ! -f "$LOG" ]]; then
  echo "Waiting for $LOG to appear..."
  while [[ ! -f "$LOG" ]]; do sleep 1; done
fi

echo "=== Waveline Monitor ==="
echo "Log: $LOG"
echo "Watching for: sessions, chunks transcribed, skips, drain events"
echo "Press Ctrl+C to stop"
echo "========================"

tail -f "$LOG" | grep --line-buffered -E \
  "Session started|Session stopped|WS transcribed|skipped transcription|draining|fully drained|WS session.*drain|timed out" | \
awk '
{
  # Extract timestamp
  ts = substr($0, 1, 20)

  # Colorize by event type
  if ($0 ~ /skipped transcription/) {
    printf "\033[31m%s\033[0m\n", $0   # red = skip
  } else if ($0 ~ /fully drained|QUEUE_DRAINED/) {
    printf "\033[32m%s\033[0m\n", $0   # green = drained
  } else if ($0 ~ /Session started/) {
    printf "\033[36m%s\033[0m\n", $0   # cyan = new session
  } else if ($0 ~ /Session stopped/) {
    printf "\033[33m%s\033[0m\n", $0   # yellow = stopped
  } else if ($0 ~ /draining/) {
    printf "\033[33m%s\033[0m\n", $0   # yellow = draining
  } else if ($0 ~ /timed out/) {
    printf "\033[35m%s\033[0m\n", $0   # magenta = warning
  } else {
    print $0                            # default = transcribed
  }
  fflush()
}'
