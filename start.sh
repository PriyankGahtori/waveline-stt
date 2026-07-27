#!/bin/bash

# Waveline STT Service Startup Script

# Ensure we are in the project directory
cd "$(dirname "$0")"

# Path to virtual environment python
PYTHON=".venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please run the installation steps from README.md first."
    exit 1
fi

echo "=== Waveline Startup ==="
echo "Which models would you like to load?"
echo "1) All local defaults (Whisper, Voxtral, Vaani) - Recommended for Apple Silicon"
echo "2) Whisper Only (Faster startup, works on all hardware)"
echo "3) Voxtral Only (Apple Silicon only)"
echo "4) Vaani Only (Hindi optimized, Apple Silicon only)"
echo "5) Nemotron Only (parakeet.cpp / GGUF)"
echo "6) Custom (Set variables manually)"
read -p "Selection [1-6]: " choice

# Default all to false, then enable based on selection
export LOAD_WHISPER=false
export LOAD_VOXTRAL=false
export LOAD_VAANI=false
export LOAD_NEMOTRON=false
export NEMOTRON_BACKEND=auto
export NEMOTRON_BIN=parakeet-cli
if [ -f "/private/tmp/parakeet_nemotron/nemotron-3.5-asr-streaming-0.6b-f16.gguf" ]; then
    export NEMOTRON_PARKEET_MODEL=/private/tmp/parakeet_nemotron/nemotron-3.5-asr-streaming-0.6b-f16.gguf
else
    export NEMOTRON_PARKEET_MODEL=nemotron-3.5-asr-streaming-0.6b
fi

case $choice in
    1)
        export LOAD_WHISPER=true
        export LOAD_VOXTRAL=true
        export LOAD_VAANI=true
        ;;
    2)
        export LOAD_WHISPER=true
        ;;
    3)
        export LOAD_VOXTRAL=true
        ;;
    4)
        export LOAD_VAANI=true
        ;;
    5)
        export LOAD_NEMOTRON=true
        export NEMOTRON_BACKEND=parakeet
        ;;
    6)
        read -p "Load Whisper? [y/N]: " w && [[ $w == "y" ]] && export LOAD_WHISPER=true
        read -p "Load Voxtral? [y/N]: " v && [[ $v == "y" ]] && export LOAD_VOXTRAL=true
        read -p "Load Vaani? [y/N]: " va && [[ $va == "y" ]] && export LOAD_VAANI=true
        read -p "Load Nemotron? [y/N]: " n && [[ $n == "y" ]] && export LOAD_NEMOTRON=true && export NEMOTRON_BACKEND=parakeet
        ;;
    *)
        echo "Invalid selection. Defaulting to Whisper only."
        export LOAD_WHISPER=true
        ;;
esac

echo "---------------------------"
echo "Starting server with:"
echo "  Whisper: $LOAD_WHISPER"
echo "  Voxtral: $LOAD_VOXTRAL"
echo "  Vaani:   $LOAD_VAANI"
echo "  Nemotron: $LOAD_NEMOTRON"
echo "  Nemotron backend: $NEMOTRON_BACKEND"
echo "  Nemotron binary:   $NEMOTRON_BIN"
echo "  Nemotron model:    $NEMOTRON_PARKEET_MODEL"
echo "---------------------------"

# Run the server
$PYTHON server.py
