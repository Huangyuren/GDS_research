#include "openslide-features.h"
#include "openslide.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <omp.h>

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }
#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }


using namespace std;
using namespace cv;


class Openslide2jpeg {
    private:
        float resize_ratio;
        vector<tuple<int64_t, int64_t, int>> target_dimension;
        vector<Mat> whole_slide_rgb;
        vector<string> slide_files;
        vector<openslide_t*> slide_obj;
    public:
        int slide_count;
        void searchPath(string);
        void loadWholeSlide();
        void trans2jpeg(int);
        void printMatrix(Mat);
};

void Openslide2jpeg::printMatrix(Mat input){
    // if(input.isContinuous()) printf("This matrix is continuous\n");
    int counter=0; // for printer limit.
    for( int y = 0; y < input.rows && counter < 1; y++ ) {
        counter++;
        for( int x = 0; x < input.cols; x++ ) {
            printf("Position [%d, %d]: , pixel value: ", y, x);
            for( int c = 0; c < input.channels(); c++ ) {
                printf("%d ", input.at<Vec3b>(y,x)[c]);
            }
            printf("\n");
        }
    } 
}

void Openslide2jpeg::searchPath(string basepath){
    DIR *dir = opendir(basepath.c_str());
    DIR *dir_inner;
    struct dirent* ent;
    struct dirent* ent_inner;
    slide_count = 0;
    if (dir != NULL) {
        //Iterate through ./data/ directory
        while ((ent = readdir(dir)) != NULL) {
            if(string(ent->d_name) != ".." && string(ent->d_name) != "."){
                //Iterator through ./data/class_*/ directory
                dir_inner = opendir((basepath + ent->d_name).c_str());
                while((ent_inner = readdir(dir_inner)) != nullptr) {
                    if(string(ent_inner->d_name) != ".." && string(ent_inner->d_name) != "."){
                        slide_files.push_back(string(basepath + string(ent->d_name) + "/" + string(ent_inner->d_name)));
                        printf ("%s\n", ent_inner->d_name);
                        slide_count += 1;
                    }
                }
                closedir(dir_inner);
            }
        }
        closedir(dir);
        printf("Total .svs files found: %d\n", slide_count);
    }
    else {
        // could not open directory
        perror ("Could not open directory.");
    }
}

void Openslide2jpeg::loadWholeSlide(){
    // printf("OpenCV version: %d.%d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    resize_ratio = 0.2;
    int level_cnt;
    for(int idx=0; idx < slide_count; idx++){
        int32_t target_level;
        int64_t target_width, target_height;
        printf("%d. Current processing file name: %s\n", idx, slide_files[idx].c_str());
        openslide_t* slide = openslide_open(slide_files[idx].c_str());
        level_cnt = openslide_get_level_count(slide);
        for(int i=0; i<level_cnt; i++){
            int32_t tmp_level = openslide_get_level_downsample(slide, i);
            if(tmp_level <= 1.0 / resize_ratio){
                target_level = i;
            }
        }
        openslide_get_level_dimensions(slide, target_level, &target_width, &target_height);
        target_dimension.push_back(make_tuple(target_width, target_height, target_level));
        slide_obj.push_back(slide);
        printf("   Target level: %d, Target width: %ld, Target height: %ld\n", target_level, target_width, target_height);
    }
    
    // uint32_t* buf;
    #pragma omp parallel num_threads(slide_count) shared(slide_obj, target_dimension)
    {
        // #pragma omp for nowait schedule(static)
        #pragma omp for 
        for(int i=0; i<slide_count; i++){
            int64_t target_width = get<0>(target_dimension[i]);
            int64_t target_height = get<1>(target_dimension[i]);
            int32_t target_level = get<2>(target_dimension[i]);
            size_t esti_size = (size_t)target_width * (size_t)target_height * (size_t)4;
            //Preparing buffer
            vector<uint32_t> curr_buf(esti_size, 0);
            // buf = reinterpret_cast<uint32_t*>(malloc((size_t)target_width * (size_t)target_height * (size_t)4));
            openslide_read_region(slide_obj[i], curr_buf.data(), 0, 0, target_level, target_width, target_height);
            //Converting buffer to OpenCV's matrix datatype
            Mat whole_slide_src = Mat(target_height, target_width, CV_8UC4, curr_buf.data());
            Mat whole_slide_rgb = Mat::zeros(target_height, target_width, CV_8UC3);
            cvtColor(whole_slide_src, whole_slide_rgb, COLOR_RGBA2RGB);
            // imwrite("test_1_wholeslide_rgba2rgb.jpg", whole_slide_rgb);

            nvjpegHandle_t nv_handle;
            nvjpegEncoderState_t nv_enc_state;
            nvjpegEncoderParams_t nv_enc_params;
            cudaStream_t stream;
            cudaStreamCreate(&stream);
             
            // initialize nvjpeg structures
            CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
            CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
            CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
            CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));
            
            printf("Iteration: %d\n", i);
            printf("   OpenMP thread count: %d\n", omp_get_num_threads());
            nvjpegImage_t nv_image;
            // Fill nv_image with image data, let’s say target_height, target_width image in RGB format
            Mat bgr[3];
            split(whole_slide_rgb, bgr);
            printf("   Target width: %ld, Target height: %ld\n", target_width, target_height);
            for(int j=0; j<3; j++){
                CHECK_CUDA(cudaMalloc((void **)&(nv_image.channel[j]), target_width * target_height));
                CHECK_CUDA(cudaMemcpy(nv_image.channel[j], bgr[2-j].data, target_width * target_height, cudaMemcpyHostToDevice));
                nv_image.pitch[j] = (size_t)target_width;
            }
            // size_t max_stream_length=0;
            // CHECK_NVJPEG(nvjpegEncodeGetBufferSize(nv_handle, nv_enc_params, target_width, target_height, &max_stream_length));
            // printf("   Max stream length estimated: %zu\n", max_stream_length);
            // Compress image
            CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                &nv_image, NVJPEG_INPUT_RGB, target_width, target_height, stream));
             
            // get compressed stream size
            size_t length;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
            // printf("Encoder gen length: %zu\n", length);
            // get stream itself
            cudaStreamSynchronize(stream);
            vector<unsigned char> jpeg(length);
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, 0));
             
            // write stream to file
            const char* jpegstream = reinterpret_cast<const char*>(jpeg.data());
            cudaStreamSynchronize(stream);
            ofstream output_file("test"+to_string(i)+".jpg", ios::out | ios::binary);
            output_file.write(jpegstream, length);
            output_file.close();
            for(int j=0; j<3; j++){
                CHECK_CUDA(cudaFree(nv_image.channel[j]));
            }
            CHECK_NVJPEG(nvjpegDestroy(nv_handle));

        }

    }
    for(int j=0; j<slide_count; j++){
        openslide_close(slide_obj[j]);
    }

}

void Openslide2jpeg::trans2jpeg(int idx){
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
     
    // initialize nvjpeg structures
    CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));
    
    printf("Iteration: %d\n", idx);
    nvjpegImage_t nv_image;
    int64_t target_width = get<0>(target_dimension[idx]);
    int64_t target_height = get<1>(target_dimension[idx]);
    // Fill nv_image with image data, let’s say target_height, target_width image in RGB format
    Mat bgr[3];
    split(whole_slide_rgb[idx], bgr);
    printf("   Whole slide RGB size: %zu, Target width: %ld, Target height: %ld\n", whole_slide_rgb.size(), target_width, target_height);
    for(int i=0; i<3; i++){
        CHECK_CUDA(cudaMalloc((void **)&(nv_image.channel[i]), target_width * target_height));
        CHECK_CUDA(cudaMemcpy(nv_image.channel[i], bgr[2-i].data, target_width * target_height, cudaMemcpyHostToDevice));
        nv_image.pitch[i] = (size_t)target_width;
    }
     
    // Compress image
    CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, NVJPEG_INPUT_RGB, target_width, target_height, stream));
     
    // get compressed stream size
    size_t length;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
    // printf("Encoder gen length: %zu\n", length);
    // get stream itself
    cudaStreamSynchronize(stream);
    vector<unsigned char> jpeg(length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, 0));
     
    // write stream to file
    const char* jpegstream = reinterpret_cast<const char*>(jpeg.data());
    cudaStreamSynchronize(stream);
    ofstream output_file("test"+to_string(idx)+".jpg", ios::out | ios::binary);
    output_file.write(jpegstream, length);
    output_file.close();
    for(int i=0; i<3; i++){
        CHECK_CUDA(cudaFree(nv_image.channel[i]));
    }
    CHECK_NVJPEG(nvjpegDestroy(nv_handle));
}

int main(int argc, char* argv[]){
    if(argc < 2){
        fprintf(stderr, "%s", "Please provide data path for OpenSlide. usage ./openslide2jpeg < slide_path > \n");
        return -1;
    }
    CHECK_CUDA(cudaSetDevice(0));
    string slide_path = argv[1];
    Openslide2jpeg slideObj;
    slideObj.searchPath(slide_path);
    slideObj.loadWholeSlide();
}
