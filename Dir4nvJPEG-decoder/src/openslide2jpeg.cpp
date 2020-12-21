#include "openslide-features.h"
#include "openslide.h"
// #include "openslide-common.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "nvjpegDecoder.h"
#include <stdint.h>
#include <stdio.h>
#include <dirent.h>
#include <vector>
#include <string>

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
        int slide_count;
        float resize_ratio;
        int64_t target_width, target_height;
        int32_t target_level;
        Mat whole_slide_rgb;
        vector<string> slide_files;
    public:
        void searchPath(string);
        void loadWholeSlide(int);
        void trans2jpeg();
};

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
        slide_count -= 1;
        printf("ToTal .svs files found: %d\n", slide_count);
    }
    else {
        // could not open directory
        perror ("Could not open directory.");
    }
}

void Openslide2jpeg::loadWholeSlide(int index){
    printf("OpenCV version: %d.%d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    resize_ratio = 0.2;
    printf("Current processing file name: %s\n", slide_files[index].c_str());
    openslide_t* slide = openslide_open(slide_files[index].c_str());
    int level_cnt = openslide_get_level_count(slide);
    printf("Total Level count: %d\n", level_cnt);
    for(int i=0; i<level_cnt; i++){
        int32_t tmp_level = openslide_get_level_downsample(slide, i);
        // printf("Iter: %d, Temporary level: %d\n", i, tmp_level);
        if(tmp_level <= 1.0 / resize_ratio){
            target_level = i;
        }
    }
    // printf("Target level: %d\n", target_level);
    openslide_get_level_dimensions(slide, target_level, &target_width, &target_height);
    printf("Target level: %d, Target width: %ld, Target height: %ld\n", target_level, target_width, target_height);
    uint32_t* buf = reinterpret_cast<uint32_t*>(malloc((size_t)target_width * (size_t)target_height * (size_t)4));
    openslide_read_region(slide, buf, 0, 0, target_level, target_width, target_height);
    Mat whole_slide_src = Mat(target_height, target_width, CV_8UC4, buf);
    whole_slide_rgb = Mat::zeros(target_height, target_width, CV_8UC3);
    cvtColor(whole_slide_src, whole_slide_rgb, COLOR_RGBA2RGB);
    // if(whole_slide_rgb.isContinuous()) printf("This matrix is continuous\n");
    // int counter=0;
    // for( int y = 0; y < whole_slide_rgb.rows && counter < 1; y++ ) {
    //     counter++;
    //     for( int x = 0; x < whole_slide_rgb.cols; x++ ) {
    //         printf("Position: (%d, %d): ", y, x);
    //         for( int c = 0; c < whole_slide_rgb.channels(); c++ ) {
    //             printf("Pixel value: %d, ", whole_slide_rgb.at<Vec3b>(y,x)[c]);
    //         }
    //         printf("\n");
    //     }
    // } 
    // imwrite("test_1_wholeslide_rgba2rgb.jpg", whole_slide_rgb);
    free(buf);
    openslide_close(slide);
}

void Openslide2jpeg::trans2jpeg(){
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
     
    nvjpegImage_t nv_image;
    // Fill nv_image with image data, let’s say target_height, target_width image in RGB format
    Mat bgr[3];
    split(whole_slide_rgb, bgr);
    for(int i=0; i<3; i++){
        CHECK_CUDA(cudaMalloc((void **)&(nv_image.channel[i]), target_width * target_height));
        CHECK_CUDA(cudaMemcpy(nv_image.channel[i], bgr[2-i].data, target_width * target_height, cudaMemcpyHostToDevice));
        nv_image.pitch[i] = (size_t)target_width;
    }
    // CHECK_CUDA(cudaMalloc((void **) &(nv_image.channel[0]), target_width * target_height * 3));
    // CHECK_CUDA(cudaMemcpy(nv_image.channel[0], whole_slide_rgb.data, target_width * target_width * 3, cudaMemcpyHostToDevice));
    // nv_image.pitch[0] = 3 * (size_t)target_width;
     
    // Compress image
    CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, NVJPEG_INPUT_RGB, target_width, target_height, stream));
     
    // get compressed stream size
    size_t length;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
    printf("Encoder gen length: %zu\n", length);
    // get stream itself
    cudaStreamSynchronize(stream);
    vector<unsigned char> jpeg(length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, 0));
     
    // write stream to file
    const char* jpegstream = reinterpret_cast<const char*>(jpeg.data());
    cudaStreamSynchronize(stream);
    ofstream output_file("test.jpg", ios::out | ios::binary);
    output_file.write(jpegstream, length);
    output_file.close();
}

int main(int argc, char* argv[]){
    if(argc < 2){
        fprintf(stderr, "%s", "Please provide data path for OpenSlide. usage ./openslide2jpeg < slide_path > \n");
        return -1;
    }
    string slide_path = argv[1];
    Openslide2jpeg slideObj;
    slideObj.searchPath(slide_path);
    slideObj.loadWholeSlide(0);
    slideObj.trans2jpeg();
}
/*def _load_img(self, index):
        #  print("Index: {}, Loading images...".format(index))
        slide = openslide.open_slide(self.file_labels[index][0])
        # Find the optimal pyramid level to read
        target_level = 0
        for level in range(len(slide.level_downsamples)):
            if slide.level_downsamples[level] <= 1.0 / self.resize_ratio:
                target_level = level

        target_width, target_height = slide.level_dimensions[target_level]
        # Read the image
        print("Index: {}, Reading images...".format(index))
        img_rgba = slide.read_region(
            location=(0, 0),
            level=target_level,
            size=(target_width, target_height),
        )
        img_rgba = np.array(img_rgba)
        img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        print("Index: {}, Finish reading images...".format(index))

        # Resize the image
        width, height = target_width, target_height
        if slide.level_downsamples[target_level] < 1.0 / self.resize_ratio:
            ratio = slide.level_downsamples[target_level] * self.resize_ratio
            width, height = int(target_width * ratio), int(target_height * ratio)
            img = cv2.resize(img, (width, height))

        # Pad and/or crop to the desired size
        delta_w = max(0, self.img_size - width)
        delta_h = max(0, self.img_size - height)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        img = img[: self.img_size, : self.img_size, :]
        print("Index: {}, Finish padding images...".format(index))
*/
