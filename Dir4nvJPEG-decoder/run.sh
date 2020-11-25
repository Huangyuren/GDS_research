#!/bin/bash

g++ -m64 ./mynvJpegDecoder.cpp -I../include -lnvjpeg -L../lib64 -I/usr/local/cuda-11.0/include -ldl -lrt -pthread -lcudart -L/usr/local/cuda-11.0/lib64 -Wl,-rpath=../lib64 -Wl,-rpath=/usr/local/cuda-11.0/lib64 -o test_out
