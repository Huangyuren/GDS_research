# GDS_research
## This repository is about my research on NVIDIA's newly released GPUDirect Storage software.
1. For more details, please refer to official documentation: [Documentation](https://docs.nvidia.com/gpudirect-storage/index.html) -> Overview guide
2. Files containing extension (.qdrep) files, which are for NVIDIA Nsight System profiler usage.
## Guide for Dir4nvJPEG-decoder
1. This folder contains experiments recently.
2. mynvJpegDecoder.cpp uses some functions in nvjpegDecoder.h which provided by nvidia, and this code is modified from nvjpegDecode.cpp which is also from nvidia.
3. Differences between mynvJpegDecoder.cpp and nvjpegDecoder.cpp are cuda streams usage. nvjpegDecode.cpp doesn't use cuda stream, so we can easily find that all decode processes are conducted by default stream. As a result, we assign cuda stream as many as user input batch size. (namely, # of cuda stream == batch size)
4. Commands for runing this code: ./run.sh (in Dir4nvJPEG-decoder), then ./build/mynvJpegDecoder -i ../data/2kResolution/ -b 5 -t 10 (-b: batch size=5, -t: total decoded images=10)
5. For more arguments indication, please refer to ![nvidia nvjpeg repo](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/nvJPEG/nvJPEG-Decoder)
