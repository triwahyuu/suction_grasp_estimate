#include "utils.hpp"
#include <string>
#include <vector>

#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

std::vector<std::string> split(std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

std::string join(std::vector<std::string> &str, const std::string& delim)
{
    std::string tokens;
    for(std::string s : str){
        if(s.find('.') != std::string::npos)
            tokens += s;
        else
            tokens += s + "/";
    }
    return tokens;
}

bool is_exist (std::string& fname)
{
  struct stat buffer;
  return (stat (fname.c_str(), &buffer) == 0); 
}


bool load_txt(std::string data_path, std::vector<std::string> &out_data)
{
    data_path = (data_path.back() != '/') ? (data_path + "/") : data_path;
    std::string fname = data_path + "test-split.txt";

    std::ifstream ifs(fname);
    if (!ifs)
        return false;

    std::string line;
    while (std::getline(ifs, line)) {
        out_data.push_back(line);
    }
    return true;
}


cv::Mat normalize_img(cv::Mat& img, cv::Scalar mean, cv::Scalar std_dev)
{
    cv::Mat ret_img = img - mean;
    cv::divide(ret_img, std_dev, ret_img);
    return ret_img;
}

bool load_dataset_prep(std::string data_path, std::string file_id, std::vector<torch::Tensor> &imgs) 
{
    data_path = (data_path.back() != '/') ? (data_path + "/") : data_path;
    std::string img_color_path = data_path + "color-input/" + file_id + ".png";
    std::string img_depth_path = data_path + "depth-input/" + file_id + ".png";

    if(is_exist(img_color_path) && is_exist(img_depth_path)){
        cv::Mat img_color, img_depth;
        cv::Scalar mean(0.485, 0.456, 0.406), std_dev(0.229,0.224,0.225);

        // color
        img_color = cv::imread(img_color_path);
        cv::cvtColor(img_color, img_color, CV_BGR2RGB);
        img_color.convertTo(img_color, CV_32FC3, 1.0f/255.0f);

        img_color = normalize_img(img_color, mean, std_dev);
        torch::Tensor color_tensor = torch::from_blob(img_color.data, 
            {1, kIMAGE_WIDTH, kIMAGE_HEIGHT, 3},
            torch::kFloat32
        );
        color_tensor = color_tensor.permute({0, 3, 1, 2});

        // depth
        img_depth = cv::imread(img_depth_path, cv::IMREAD_UNCHANGED);
        img_depth.convertTo(img_depth, CV_32FC3, 1.0f/8000.0f);
        cv::Mat depth_split[3] = {img_depth, img_depth, img_depth};
        
        cv::Mat img_ddd;
        cv::merge(depth_split, 3, img_ddd);
        img_ddd = normalize_img(img_ddd, mean, std_dev);
        torch::Tensor ddd_tensor = torch::from_blob(img_ddd.data, 
            {1, kIMAGE_WIDTH, kIMAGE_HEIGHT, 3},
            torch::kFloat32
        );
        ddd_tensor = ddd_tensor.permute({0, 3, 1, 2});

        imgs.push_back(color_tensor);
        imgs.push_back(ddd_tensor);

    }
    else{
        std::cerr << "data file_id not exist" << std::endl;
        return false;
    }
    return true;
}

bool load_dataset(std::string data_path, std::string file_id, std::vector<cv::Mat> &imgs)
{
    data_path = (data_path.back() != '/') ? (data_path + "/") : data_path;
    std::string img_color_path = data_path + "color-input/" + file_id + ".png";
    std::string img_depth_path = data_path + "depth-input/" + file_id + ".png";

    if(is_exist(img_color_path) && is_exist(img_depth_path)){
        cv::Mat color_img, depth_img, ddd_img;

        // color
        color_img = cv::imread(img_color_path);
        cv::cvtColor(color_img, color_img, CV_BGR2RGB);
        color_img.convertTo(color_img, CV_32FC3, 1.0f/255.0f);

        // depth
        depth_img = cv::imread(img_depth_path, cv::IMREAD_UNCHANGED);
        depth_img.convertTo(depth_img, CV_32FC3, 1.0f/8000.0f);
        cv::Mat depth_split[3] = {depth_img, depth_img, depth_img};
        cv::merge(depth_split, 3, ddd_img);
        

        imgs.push_back(color_img);
        imgs.push_back(ddd_img);
    }
    else{
        std::cerr << "data file_id not exist" << std::endl;
        return false;
    }
    return true;
}

bool prep_dataset(std::vector<cv::Mat> &imgs, std::vector<torch::Tensor> &out_tensor)
{
    struct torch::data::transforms::Normalize<torch::Tensor> 
        normalize_tensor({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    return true;
}