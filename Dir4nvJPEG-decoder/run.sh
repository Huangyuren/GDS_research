#!/bin/bash

DIR="./build/"
if [ -d "$DIR" ]; then
    echo "Rebuilding nvjpeg code in ${DIR}..."
    rm -rf ./build
else
    echo "Building nvjpeg code in ${DIR}..."
fi

mkdir build && cd build
cmake ../
make
