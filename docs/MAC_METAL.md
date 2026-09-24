# Running on an Apple Silicon Mac (Metal)

The README's "Setting up a Mac" section is the walkthrough for lab members; this file holds the
technical detail behind it.

The pipeline has one accelerated stage, Cellpose-SAM nucleus segmentation, which runs through PyTorch.
On an Apple Silicon Mac (M1 to M4) PyTorch drives the GPU through Metal Performance Shaders (MPS);
everything else in the pipeline is NumPy, SciPy, scikit-image and SpotMAX on the CPU on every platform.
Device choice lives in `src/germquant/device.py`: CUDA when present, else Metal, else CPU, overridable
with `GERMQUANT_DEVICE=cuda|mps|cpu` or `segmentation.nuclei.device` in a profile. `germquant check-gpu`
reports which one a machine will use.

## Setting up (one time)

Requirements: macOS 14 or newer, an Apple Silicon Mac with 16 GB or more of unified memory (a
full-resolution gonad needs several GB), about 8 GB of disk, Python 3.11.

1. Install Python 3.11 from python.org (the macOS installer) or with Homebrew (`brew install python@3.11`).
2. Install `uv`: `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh` and reopen Terminal.
3. Get this code (download the ZIP from GitHub and unzip it, or `git clone` it) and open Terminal in the folder.
4. Run, one at a time:
   ```
   uv venv --python 3.11
   uv pip install -c constraints/mac-arm64-2026-09.txt -e ".[gpu,sc]"
   uv pip install -c constraints/mac-arm64-2026-09.txt spotmax cellacdc
   ```
   torch is installed from PyPI; its macOS arm64 wheel carries Metal support. Do not use the
   `download.pytorch.org/whl/cu128` index from the Windows instructions (there is no CUDA on a Mac).
5. Fetch the trained nucleus model: `.venv/bin/germquant get-model` downloads the GitHub release asset
   to `models/models/germline_nuclei_combined` and verifies its sha256 (set `GERMQUANT_MODELS` to put
   it elsewhere).
6. Check the GPU is seen: `.venv/bin/germquant check-gpu` should print a line containing
   "Apple Silicon GPU via Metal" and a line starting with "OK".
7. `chmod +x quantify.command` once, then double-click `quantify.command` (first time: the once-only
   macOS check), drag an `.nd2` into the Terminal window and press Return; or run
   ```
   .venv/bin/germquant run "/path/to/IMAGE.nd2" --config config/config.yaml --out "/path/to/RESULTS"
   ```
   Every command in the README works the same on a Mac with `.venv/bin/germquant` in place of
   `.venv\Scripts\germquant.exe` and forward slashes in paths. Dropping an image onto the
   `quantify.command` icon in Finder does nothing (a `.command` file is a document, not an
   application); the script asks for the image after it opens.

## What to expect

* Speed: Metal is slower than the RTX 5090 for the Cellpose pass (expect the segmentation of a
  full-resolution gonad to take several times longer; the CPU stages take about the same time as on
  the workstation per core). Use `--xy-stride 2` for a quick look, never for numbers.
* Precision: Cellpose runs in float32 on Metal (bfloat16 support there is incomplete; the workstation
  uses bfloat16 on CUDA). Nucleus masks can therefore differ slightly between a Mac and the
  workstation. Before pooling Mac-produced numbers with workstation numbers, run one of the golden
  gonads on the Mac and compare with `scripts/regression_diff.py` (label-id permutations pass; a
  different mask does not).
* Unsupported operators: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically when Metal is chosen,
  so an operator Metal cannot run falls back to the CPU instead of stopping the run. If Cellpose fails
  on Metal outright (for example out of memory on a huge stack), the log says "Cellpose failed on mps
  ... retrying on the CPU", the same model is retried once on the CPU before the classical watershed
  fallback is considered, and the `segmentation_method` column still says which segmenter produced
  the labels.
* SpotMAX (RAD-51 counting) is CPU-only on every platform; CuPy is not installed on the Mac.
* Not applicable on a Mac: the Windows long-path handling (a no-op elsewhere), `quantify.bat`,
  the Docker / Apptainer images (Linux, CUDA).

## Status

The Metal path was written on the Windows workstation from the Cellpose and PyTorch documentation and
is covered by unit tests that fake the accelerators; it has not yet been run on a real Mac. The first
Mac run should be `germquant check-gpu`, then one golden gonad with the regression diff, and the
README's Mac times should be replaced by measured ones.
