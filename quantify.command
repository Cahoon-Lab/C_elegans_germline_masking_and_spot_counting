#!/bin/bash
# ============================================================================
#  quantify.command - count RAD-51 spots in ONE .nd2 image on a Mac. No coding needed.
#
#  HOW TO USE (two ways):
#    1) Drag your .nd2 file ONTO this file's icon in Finder, OR
#    2) Double-click this file, then drag your .nd2 into the Terminal window and press Enter.
#  Your results appear in a new folder next to your image (named *_results).
#  First time only: right-click the file, choose Open, and confirm (macOS asks once for a script
#  downloaded from the internet); `chmod +x quantify.command` if it does not run.
# ============================================================================
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo
  echo "ERROR: the software environment is not set up (the .venv folder is missing)."
  echo "Ask whoever set up this computer, or see the README section"
  echo '  "Setting up an Apple Silicon Mac (one time)".'
  echo
  read -r -p "Press Enter to close. "
  exit 1
fi
IMG="$1"
if [ -z "$IMG" ]; then
  read -r -p "Drag your .nd2 image into this window, then press Enter: " IMG
fi
IMG="${IMG%\'}"; IMG="${IMG#\'}"; IMG="${IMG// /\ }"; IMG="$(printf '%s' "$IMG" | sed "s/\\\\ / /g")"
if [ ! -f "$IMG" ]; then
  echo
  echo "ERROR: that file was not found:"
  echo "  $IMG"
  echo "Make sure you dragged a real .nd2 file in."
  echo
  read -r -p "Press Enter to close. "
  exit 1
fi
OUT="${IMG%.nd2}_results"
echo
echo "============================================================"
echo " Quantifying RAD-51 spots"
echo " Image:    $IMG"
echo " Results:  $OUT"
echo " On an Apple Silicon Mac this takes longer than on the lab PC. Leave this window open."
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
read -r -p "Press Enter to close. "
