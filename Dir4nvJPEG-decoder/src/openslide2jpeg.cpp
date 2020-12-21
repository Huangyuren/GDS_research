#include "openslide-features.h"
#include "openslide.h"
// #include "openslide-common.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "nvjpegDecoder.h"
#include <stdint.h>
#include <stdio.h>
#include <dirent.h>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class Openslide2jpeg {
    private:
        int slide_count;
        float resize_ratio;
        int64_t target_width, target_height;
        int32_t target_level;
        uint32_t* buf;
        vector<string> slide_files;
    public:
        void searchPath(string);
        void loadWholeSlide(int);
};

void Openslide2jpeg::searchPath(string basepath){
    DIR *dir = opendir(basepath.c_str());
    DIR *dir_inner;
    struct dirent* ent;
    struct dirent* ent_inner;
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
    printf("Target level: %d, Target width: %ld, Target height: %ld", target_level, target_width, target_height);
    buf = reinterpret_cast<uint32_t*>(malloc((size_t)target_width * (size_t)target_height * (size_t)4));
    openslide_read_region(slide, buf, 0, 0, target_level, target_width, target_height);
    Mat whole_slide_src = Mat(target_height, target_width, CV_8UC4, buf);
    Mat whole_slide_rgb = Mat::zeros(target_height, target_width, CV_8UC3);
    cvtColor(whole_slide_src, whole_slide_rgb, COLOR_BGRA2RGB);
    imwrite("test_1_wholeslide.jpg", whole_slide_rgb);
    free(buf);
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
