#include "OpenCvTest.h"
#include <random>
void OpenCv::DisplayImage(cv::Mat & image)
{
	if (!image.empty())
	{
		cv::namedWindow("Display");
		cv::imshow("Display", image);
		cv::waitKey(0);
	}
}

void OpenCv::WhiteNoiseImage(cv::Mat & image, int noise)
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
	std::uniform_int_distribution<int> randomCol(0, image.cols- 1);
	size_t row, col;
	for (size_t idx = 0; idx < noise;idx++)
	{
		row = randomRow(generator);
		col = randomCol(generator);
		if (image.type() == CV_8UC1)
		{
			image.at<uchar>(row,col) = 255;
		}
		else
		{
			image.at<cv::Vec3b>(row, col)[0] = 255;
			image.at<cv::Vec3b>(row, col)[1] = 255;
			image.at<cv::Vec3b>(row, col)[2] = 255;
		}
	}
}
