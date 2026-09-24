#!/bin/bash
# ============================================================================
#  quantify.command - count RAD-51 spots in ONE .nd2 image on a Mac. No coding needed.
#
#  HOW TO USE: double-click this file. A Terminal window opens and asks for the image.
#  Drag your .nd2 file from Finder into that window, then press Return.
#  (Dropping the image onto this file's icon does nothing on a Mac; the file has to be opened first.)
#  Your results appear in a new folder next to your image (named *_results).
#  First time only: macOS refuses files downloaded from the internet once. macOS 15 or newer:
#  double-click, click Done, then System Settings > Privacy & Security > Open Anyway, then
#  double-click again. macOS 14: right-click the file, choose Open, click Open.
#  If Terminal says you do not have appropriate access privileges: in Terminal, in this folder,
#  run  chmod +x quantify.command  once.
# ============================================================================
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo
  echo "ERROR: the software environment is not set up (the .venv folder is missing)."
  echo "Ask whoever set up this computer, or see the README section"
  echo '  "Setting up a Mac".'
  echo
  read -r -p "Press Return to finish, then close this window (Command-W). "
  exit 1
fi
IMG="$1"
if [ -z "$IMG" ]; then
  read -r -p "Drag your .nd2 image into this window, then press Return: " IMG
fi
# Terminal pastes a dragged path with a backslash before every space, parenthesis or quote, and
# sometimes a trailing space; strip those (remove-only, so a plain path is left exactly as it is).
IMG="${IMG% }"
IMG="${IMG%\'}"; IMG="${IMG#\'}"
IMG="$(printf '%s' "$IMG" | sed 's/\\\(.\)/\1/g')"
if [ ! -f "$IMG" ]; then
  echo
  echo "ERROR: that file was not found:"
  echo "  $IMG"
  echo "Make sure you dragged a real .nd2 file in."
  echo
  read -r -p "Press Return to finish, then close this window (Command-W). "
  exit 1
fi
OUT="$(printf '%s' "$IMG" | sed -E 's/\.[nN][dD]2$//')_results"
echo
echo "============================================================"
echo " Quantifying RAD-51 spots"
echo " Image:    $IMG"
echo " Results:  $OUT"
echo " On a Mac this takes longer than on the lab workstation. Leave this window open."
echo "============================================================"
echo
export PYTHONIOENCODING=utf-8
"$PY" -m germquant.cli run "$IMG" --config "config/config.yaml" --out "$OUT"
status=$?
echo
if [ $status -ne 0 ]; then
  echo "*** Something went wrong - read the messages above, then check the"
  echo "    README section \"If something breaks\". ***"
else
  echo "*** DONE! Your results are in: ***"
  echo "    $OUT"
  echo
  echo "Open the file that ends in \"__nuclei.csv\" in Excel or Numbers."
  echo "The \"n_spots\" column is the RAD-51 count for each nucleus."
fi
echo
read -r -p "Press Return to finish, then close this window (Command-W). "
