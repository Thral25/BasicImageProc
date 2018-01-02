#pragma once
#include <opencv2\opencv.hpp>
class OpenCv
{
public:
	static void DisplayImage(cv::Mat& image);
	static void WhiteNoiseImage(cv::Mat& image,int noise);
};