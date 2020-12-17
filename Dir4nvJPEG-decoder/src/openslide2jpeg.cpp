#include "openslide-features.h"
#include "nvjpegDecoder.h"
#include <stdint.h>
#include <stdio.h>
#include <dirent.h>
#include <vector>
#include <string>

using namespace std;

class Openslide2jpeg {
    private:
        int slide_count;
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
                dir_inner = opendir((basepath + "/" + ent->d_name).c_str());
                while((ent_inner = readdir(dir_inner)) != nullptr) {
                    slide_files.push_back(string(basepath + "/" + string(ent_inner->d_name)));
                    if(string(ent_inner->d_name) != ".." && string(ent_inner->d_name) != "."){
                        printf ("%s\n", ent_inner->d_name);
                        slide_count += 1;
                    }
                }
                closedir(dir_inner);
            }
        }
        closedir(dir);
        printf("ToTal .svs files found: %d\n", slide_count);
    }
    else {
        // could not open directory
        perror ("Could not open directory.");
    }
}


void Openslide2jpeg::loadWholeSlide(int index){
    openslide_t* slide = openslide_open(slide_files[index]);
    int level_cnt = openslide_get_level_count(slide);
    printf("Level count: %d\n", level_cnt);
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
