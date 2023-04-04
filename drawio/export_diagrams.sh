#!/bin/bash

# This script exports all diagrams in the current directory to PNG files.
# It requires the draw.io CLI to be installed.
mkdir -p images

for f in *.drawio; do
    drawio -x -f png --scale 2.0 -o images/${f%}.png $f
done
