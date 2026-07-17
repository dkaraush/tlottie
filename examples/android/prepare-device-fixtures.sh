#!/usr/bin/env bash
set -euo pipefail

ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Android/Sdk}}"
ADB="${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}"
SERIAL="${1:-RZCX41P9KTM}"
PACKAGE="${PACKAGE:-com.example.tlottie}"
HOST_FIXTURES="${HOST_FIXTURES:-$HOME/Documents/fixtures-full}"
FIXTURES="${DEVICE_FIXTURES:-/storage/emulated/0/Android/data/$PACKAGE/files/tgs_dump}"

if [[ ! -d "$HOST_FIXTURES" ]]; then
    echo "Fixture directory not found on host: $HOST_FIXTURES" >&2
    exit 1
fi

# App-specific external storage remains readable under scoped storage and newer
# SELinux policies that prevent apps from opening /data/local/tmp.
if ! "$ADB" -s "$SERIAL" shell test -d "$FIXTURES"; then
    "$ADB" -s "$SERIAL" shell mkdir -p "$FIXTURES"
    "$ADB" -s "$SERIAL" push "$HOST_FIXTURES/." "$FIXTURES/"
fi
"$ADB" -s "$SERIAL" shell chmod -R o+rX "$FIXTURES"

COUNT=$("$ADB" -s "$SERIAL" shell "find '$FIXTURES' -type f -name '*.json' | wc -l" | tr -d '\r')
echo "Prepared $COUNT fixtures from $FIXTURES for the tlottie debug selector."
