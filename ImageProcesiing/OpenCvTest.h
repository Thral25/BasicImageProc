#pragma once
#include <opencv2\opencv.hpp>
class OpenCv
{
public:
	static void DisplayImage(cv::Mat& image);
	static void WhiteNoiseImage(cv::Mat& image,int noise);
	static void ColorReduce(cv::Mat image, int factor);
	static void ColorReduceIter(cv::Mat image, int factor);
	static void ColorReduceMask(cv::Mat image, int factor);
	static void SharpenImage(const cv::Mat& image, cv::Mat& result);
	static void SharpenImageWithKernel(const cv::Mat& image, cv::Mat& result);
};