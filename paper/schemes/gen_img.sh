#!/bin/bash

cd $HOME/pcflow/paper/schemes/

for f in *.drawio; do
    drawio -x -f png --scale 2.0 --border 20 \
        -o img/${f%.*}.png $f
done

cd $HOME/pcflow/


