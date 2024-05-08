#!/bin/bash

rm -rf .ruff_cache
rm -rf .venv

# Remove all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} \;