#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
if [ "$1" = "register" ]; then
  "$PYTHON" register_face.py --demo
elif [ "$1" = "recognize" ]; then
  "$PYTHON" recognize_faces.py --demo
else
  echo "Usage: ./run.sh [register|recognize]"
  exit 1
fi
