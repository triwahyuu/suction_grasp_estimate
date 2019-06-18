#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<std::string> split(std::string&, const std::string&);
std::string join(std::vector<std::string>&, const std::string&);
bool is_exist (std::string& fname);

cv::Mat normalize_data(cv::Mat& input_img, cv::Scalar mean, cv::Scalar std_dev);