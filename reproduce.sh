#!/bin/bash
# Reproduce all results from the paper
# "The Endogenous Grid Method for Epstein-Zin Preferences"

set -e

cd "$(dirname "$0")"

usage() {
    echo "Usage: $0 [-h|--help] [-y|--yes] [--paper] [--results] [--all]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -y, --yes      Auto-confirm uv installation (for non-interactive use)"
    echo "  --paper        Build the PDF paper"
    echo "  --results      Run benchmarks and generate figures"
    echo "  --all          Run both (default if no option given)"
    exit "${1:-0}"
}

auto_yes=false
run_paper=false
run_results=false

# Parse arguments first
while [[ $# -gt 0 ]]; do
    case $1 in
        --paper)
            run_paper=true
            shift
            ;;
        --results)
            run_results=true
            shift
            ;;
        --all)
            run_paper=true
            run_results=true
            shift
            ;;
        -y|--yes)
            auto_yes=true
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1"
            usage 1
            ;;
    esac
done

# Default to --all if no task specified
if ! $run_paper && ! $run_results; then
    run_paper=true
    run_results=true
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv is required but not installed."

    if $auto_yes; then
        echo "Installing uv (--yes flag set)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    elif [[ -t 0 ]]; then
        read -p "Install uv now? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
        else
            echo "Please install uv from https://docs.astral.sh/uv/ and try again."
            exit 1
        fi
    else
        echo "Non-interactive mode: use -y to auto-install uv, or install manually."
        echo "See https://docs.astral.sh/uv/"
        exit 1
    fi
fi

echo "Installing dependencies..."
uv sync

if $run_results; then
    echo "Running benchmarks..."
    uv run python code/benchmark_paper.py
    echo "Results saved to content/figures/"
fi

if $run_paper; then
    echo "Building PDF..."
    uv run myst build --pdf
    echo "Paper saved to content/ezegm.pdf"
fi

echo "Done."
