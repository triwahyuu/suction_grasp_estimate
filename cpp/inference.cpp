#include <torch/script.h>
#include <torch/torch.h>
#include "utils.hpp"

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_WIDTH    640
#define kIMAGE_HEIGHT   480
#define kCHANNELS       3

bool load_dataset(std::string data_path, std::string file_id, std::vector<cv::Mat> &imgs) {
    data_path = (data_path.back() != '/') ? (data_path + "/") : data_path;
    std::string img_color_path = data_path + "color-input/" + file_id + ".png";
    std::string img_depth_path = data_path + "depth-input/" + file_id + ".png";

    if(is_exist(img_color_path) && is_exist(img_depth_path)){
        cv::Mat img_color, img_depth;
        cv::Scalar mean(0.485, 0.456, 0.406), std_dev(0.229,0.224,0.225);

        img_color = cv::imread(img_color_path);
        cv::cvtColor(img_color, img_color, CV_BGR2RGB);
        img_color.convertTo(img_color, CV_32FC3, 1.0f/255.0f);
        img_color = normalize_data(img_color, mean, std_dev);

        img_depth = cv::imread(img_depth_path, cv::IMREAD_UNCHANGED);
        img_depth.convertTo(img_depth, CV_32FC3, 1.0f/8000.0f);
        cv::Mat depth_split[3] = {img_depth, img_depth, img_depth};
        cv::Mat img_ddd;
        cv::merge(depth_split, 3, img_ddd);

        imgs.push_back(img_color);
        imgs.push_back(img_ddd);
    }
    else{
        std::cerr << "data file_id not exist" << std::endl;
        return false;
    }
    
    // image = cv::imread(data_path + "color" + file_id);  // CV_8UC3
    // if (image.empty() || !image.data) {
    //     return false;
    // }

    // cv::cvtColor(image, image, CV_BGR2RGB);
    // image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    return true;
}

int main(int argc, const char *argv[]) 
{
    if (argc != 3) {
        // std::cerr << "Usage: inference <path-to-model> <path-to-dataset> <img-id>"
        std::cerr << "Usage: inference <path-to-dataset> <img-id>"
                << std::endl;
        return -1;
    }
    torch::manual_seed(1234);
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Using GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Using CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::string dataset_path(argv[1]);
    std::string fileid(argv[2]);

    std::vector<cv::Mat> img_data;
    load_dataset(dataset_path, fileid, img_data);
    torch::Tensor color_img = torch::from_blob(img_data[0].data, 
        {1, kIMAGE_WIDTH, kIMAGE_HEIGHT, 3},
        torch::kFloat32
    );
    torch::Tensor ddd_img = torch::from_blob(img_data[1].data, 
        {1, kIMAGE_WIDTH, kIMAGE_HEIGHT, 3},
        torch::kFloat32
    );
    color_img = color_img.permute({0, 3, 1, 2});
    ddd_img = ddd_img.permute({0, 3, 1, 2});
    std::cout << color_img.sizes() << std::endl;

    // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
    // module->to(at::kCUDA); // to GPU
    // assert(module != nullptr);
    // std::cout << "== Model loaded!\n";

    // std::vector<std::string> labels;
    // if (LoadImageNetLabel(argv[2], labels)) {
    //     std::cout << "== Label loaded! Let's try it\n";
    // } else {
    //     std::cerr << "Please check your label file path." << std::endl;
    //     return -1;
    // }

    // // std::string file_name = "";
    // cv::Mat image;
    // if (LoadImage(argv[3], image)) {
    //     auto input_tensor = torch::from_blob(image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
    //     input_tensor = input_tensor.permute({0, 3, 1, 2});
    //     input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    //     input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    //     input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

    //     // to GPU
    //     input_tensor = input_tensor.to(at::kCUDA);

    //     torch::Tensor out_tensor = module->forward({input_tensor}).toTensor();

    //     auto results = out_tensor.sort(-1, true);
    //     auto softmaxs = std::get<0>(results)[0].softmax(0);
    //     auto indexs = std::get<1>(results)[0];

    //     for (int i = 0; i < kTOP_K; i++) {
    //         auto idx = indexs[i].item<int>();
    //         std::cout << "    ============= Top-" << i + 1
    //                     << " =============" << std::endl;
    //         std::cout << "    Label:  " << labels[idx] << std::endl;
    //         std::cout << "    With Probability:  "
    //                     << softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
    //     }
    //     return 0;
    // } 
    // else {
    //     std::cout << "Can't load the image, please check your path." << std::endl;
    //     return -1;
    // }
    
}