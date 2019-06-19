#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#define kIMAGE_WIDTH    640
#define kIMAGE_HEIGHT   480
#define kCHANNELS       3

std::vector<std::string> split(std::string&, const std::string&);
std::string join(std::vector<std::string>&, const std::string&);
bool is_exist (std::string& fname);

bool load_txt(std::string data_path, std::vector<std::string> &out_data);

cv::Mat normalize_img(cv::Mat& input_img, cv::Scalar mean, cv::Scalar std_dev);
bool load_dataset_prep(std::string data_path, std::string file_id, std::vector<torch::Tensor> &tensor_out);
bool load_dataset(std::string data_path, std::string file_id, std::vector<cv::Mat> &imgs_output);
bool prep_dataset(std::vector<cv::Mat> &input_imgs, std::vector<torch::Tensor> &out_tensor);