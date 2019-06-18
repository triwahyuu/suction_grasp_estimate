#include <string>
#include <vector>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

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

cv::Mat normalize_data(cv::Mat& img, cv::Scalar mean, cv::Scalar std_dev)
{
    cv::Mat ret_img = img - mean;
    cv::divide(ret_img, std_dev, ret_img);
    return ret_img;
}