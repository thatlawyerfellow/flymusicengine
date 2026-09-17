#!/bin/sh
# Install native dependencies entirely within the existing project venv (macOS arm64).
set -eu
cd "$(dirname "$0")/.."
test -d .venv || python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sunset.txt
mkdir -p .venv/native-bootstrap soundfonts
if ! test -x .venv/native/bin/fluidsynth; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) platform=osx-arm64 ;;
    Linux-x86_64) platform=linux-64 ;;
    *) echo 'Install FluidSynth and FFmpeg in a local environment for this platform.' >&2; exit 1 ;;
  esac
  curl -fsSL "https://micro.mamba.pm/api/micromamba/$platform/latest" -o .venv/native-bootstrap/micromamba.tar.bz2
  tar -xjf .venv/native-bootstrap/micromamba.tar.bz2 -C .venv/native-bootstrap bin/micromamba
  MAMBA_ROOT_PREFIX="$PWD/.venv/mamba" .venv/native-bootstrap/bin/micromamba create -y -p "$PWD/.venv/native" -c conda-forge fluidsynth ffmpeg
fi
.venv/bin/python scripts/download_sunset_fonts.py
