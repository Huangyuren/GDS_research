# Required files
SRC := ./src/
BIN := ./bin/
FILE2COMPILE := openslide2jpeg

SOURCE := $(SRC)$(FILE2COMPILE).cpp
TARGET := $(BIN)$(FILE2COMPILE)

# Compilers
CC=g++

# Libraries and flags
LDFLAGS=-L../lib64 -L/usr/local/cuda-11.0/lib64 -L/usr/local/lib
INCLUDES=-I./include -I/usr/local/cuda-11.0/include -I../../openslide/src -I../../openslide/common -I/usr/local/include/opencv4
LIBS=-ldl -lrt -lcudart -lnvjpeg -lopenslide -lopencv_core -lopencv_imgcodecs
RPATH=-Wl,-rpath=../lib64 -Wl,-rpath=/usr/local/cuda-11.0/lib64
CXXFLAGS := -O2 -std=c++14 $(LDFLAGS) $(INCLUDES) $(LIBS) $(RPATH)

# Rules
all: cuda
cuda: $(SOURCE)
		$(CC) -m64 $(SOURCE) $(CXXFLAGS) -o $(TARGET) $(OPENCV_LIBS)
.PHONY: clean

clean:
		rm -f $(BIN)$(FILE2COMPILE)
