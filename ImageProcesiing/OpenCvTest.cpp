#include "OpenCvTest.h"

void OpenCv::DisplayImage(cv::Mat & image)
{
	if (!image.empty())
	{
		cv::namedWindow("Display");
		cv::imshow("Display", image);
		cv::waitKey(0);
	}
}
