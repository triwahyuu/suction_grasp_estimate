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
#include <chrono>

int main(int argc, const char *argv[]) 
{
    if (argc != 4) {
        std::cerr << "Usage: inference <path-to-model> <path-to-dataset> <img-id>"
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
    torch::Device device(torch::kCPU);

    std::string model_path(argv[1]);
    std::string dataset_path(argv[2]);
    std::string fileid(argv[3]);

    // load data
    std::vector<torch::Tensor> img_data;
    load_dataset_prep(dataset_path, fileid, img_data);
    auto color_input = img_data[0], ddd_input = img_data[1];
    color_input = color_input.to(device);
    ddd_input = ddd_input.to(device);
    std::cout << "== Data loaded!\n";

    struct torch::data::transforms::Normalize<torch::Tensor> 
        normalize_tensor({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});

    // load model
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path, device);
    module->eval();
    assert(module != nullptr);
    std::cout << "== Model loaded!\n";

    // inference
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor out_tensor = module->forward({color_input, ddd_input}).toTensor();
    auto end = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "inference time:\t" << infer_duration.count() << " ms\n";

    auto cls_pred = out_tensor.argmax(1).to(torch::kFloat32);
    auto aff_pred = out_tensor.to(torch::kFloat32)[0][1];
    // std::cout << aff_pred.min() << "  " << aff_pred.max() << std::endl;
    
    // aff_pred = torch::upsample_bilinear2d(aff_pred, {kIMAGE_HEIGHT, kIMAGE_WIDTH}, false);
    aff_pred = (aff_pred - aff_pred.min())/(aff_pred.max() - aff_pred.min());

    // std::cout << aff_pred.sizes() << std::endl;
    // std::cout << aff_pred.min() << "  " << aff_pred.max() << std::endl;

    cv::Mat aff_img(kIMAGE_WIDTH, kIMAGE_HEIGHT, CV_8UC1);
    torch::Tensor out_img = aff_pred.mul(255).clamp(0, 255).to(torch::kU8);
    std::memcpy((void*)aff_img.data, out_img.data_ptr(),sizeof(torch::kU8)*out_img.numel());
    cv::resize(aff_img, aff_img, cv::Size2i(kIMAGE_WIDTH, kIMAGE_HEIGHT));

    cv::namedWindow("output data", cv::WINDOW_AUTOSIZE);
    cv::imshow("output data", aff_img);
    cv::waitKey(0);

    return 0;
}