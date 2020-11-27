#!/bin/bash

file_name="mynvJpegDecoder"
VAR1="${file_name}.cpp"
g++ -m64 ./$VAR1 -I../include -lnvjpeg -L../lib64 -I/usr/local/cuda-11.0/include -ldl -lrt -pthread -lcudart -L/usr/local/cuda-11.0/lib64 -Wl,-rpath=../lib64 -Wl,-rpath=/usr/local/cuda-11.0/lib64 -o $file_name
