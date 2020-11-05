#!/bin/bash

timestamp=$(date +"%F_%T")
echo $timestamp
# timestamp=$(echo $timestamp | sed 's/\//\-/g' $timestamp)
touch log_strace/trace_$timestamp\.txt
strace python3 ./vgg.py > log_strace/trace_$timestamp\.txt 2>&1
