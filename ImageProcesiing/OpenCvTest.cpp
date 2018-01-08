#include "OpenCvTest.h"
#include <random>
void OpenCv::DisplayImage(cv::Mat & image)
{
	if (!image.empty())
	{
		cv::namedWindow("Display", cv::WINDOW_KEEPRATIO);
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
		else if(image.type() == CV_8UC3)
		{
			image.at<cv::Vec3b>(row, col)[0] = 255;
			image.at<cv::Vec3b>(row, col)[1] = 255;
			image.at<cv::Vec3b>(row, col)[2] = 255;
		}
	}
}

void OpenCv::ColorReduce(cv::Mat image, int factor)
{
	int rows = image.rows;
	int width = image.cols*image.channels();
	for (int j = 0; j < rows; j++)
	{
		uchar* data = image.ptr<uchar>(j);
		for (int i = 0; i < width; i++)
		{
			data[i] = data[i] / factor * factor + factor / 2;
		}
	}
}

void OpenCv::ColorReduceIter(cv::Mat image, int factor)
{
	int n = static_cast<int>(log(static_cast<double>(factor)) / log(2.0) + 0.5);
	uchar mask = 0XFF << n;
	uchar factor2 = factor >> 1;

	cv::MatIterator_<cv::Vec3b> it=image.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itEnd = image.end<cv::Vec3b>();
	for (; it != itEnd; ++it)
	{
		(*it)[0]&= mask;
		(*it)[0] += factor2;
		(*it)[1] &= mask;
		(*it)[1] += factor2;
		(*it)[2] &= mask;
		(*it)[2] += factor2;

	}
}

void OpenCv::ColorReduceMask(cv::Mat image, int factor)
{
	int rows = image.rows;
	int width = image.cols*image.channels();
	if (image.isContinuous())
	{
		width = rows * width;
		rows = 1;
	}
	int n = static_cast<int>(log(static_cast<double>(factor)) / log(2.0) + 0.5);
	uchar mask = 0XFF << n;
	uchar factor2 = factor >> 1;
	for (int j = 0; j < rows; j++)
	{
		uchar* data = image.ptr<uchar>(j);
		for (int i = 0; i < width; i++)
		{
			*data &= mask;
			*data++ += factor2;
		}
	}
}

void OpenCv::SharpenImage(const cv::Mat & image, cv::Mat & result)
{
	int height = image.rows;
	int nChannels = image.channels();
	result.create(height, image.cols, image.type());
	for (int j = 1; j < height - 1; j++)
	{
		const uchar* prev = image.ptr<const uchar>(j - 1);
		const uchar* cur = image.ptr<const uchar>(j);
		const uchar* next	 = image.ptr<const uchar>(j + 1);
		uchar* res = result.ptr<uchar>(j);
		for (int i = nChannels; i < (image.cols - 1)*nChannels; i++)
		{
			*res++ = cv::saturate_cast<uchar>(5 * cur[i] - cur[i - nChannels] - cur[i + nChannels] - prev[i] - next[i]);
		}
	}
	result.row(0).setTo(cv::Scalar(0,0,0));
	result.row(result.rows-1).setTo(cv::Scalar(0, 0, 0));
	result.col(0).setTo(cv::Scalar(0, 0, 0));
	result.col(result.cols- 1).setTo(cv::Scalar(0, 0, 0));
}

void OpenCv::SharpenImageWithKernel(const cv::Mat & image, cv::Mat & result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5;
	kernel.at<float>(0, 1) = -1;
	kernel.at<float>(2, 1) = -1;
	kernel.at<float>(1, 0) = -1;
	kernel.at<float>(1, 2) = -1;
	cv::filter2D(image, result, image.depth(), kernel);
}
