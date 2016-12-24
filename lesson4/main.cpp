///@File: main.cpp
///@Brief: Contains interface of console application for testing background
///        subtraction algorithms
///@Author: Vitaliy Baldeev
///@Date: 01 October 2015

#include <iostream>
#include <fstream>
#include <memory>
#include <list>

#include "opencv2\video\video.hpp"
#include "opencv2\highgui\highgui.hpp"

#include "SegmentMotionDiff.h"
#include "SegmentMotionBU.h"
#include "SegmentMotionGMM.h"
#include "SegmentMotion1G.h"

void wrFile(std::vector<double>& obj, const std::string& fileName, int mode, const std::string& mess)
{
	std::ofstream out(fileName, mode);
	if (!out)
	{
		std::cout << "Cannot open file.\n";
	}
	out << mess + " ";
	for (const auto& val : obj)
	{
		out << val << " ";
	}
	out << std::endl;

	out.close();
}

//output: [0] - F;[1] - Precision;[2] - Recall;[3] - Specificity;[4] - False Positive Rate;
//[5] - False Negative Rate;[6] - Percentage of Wrong Classifications;
std::vector<double> F_Measure(cv::Mat& accurImage, cv::Mat& approxImage)
{
	std::vector<double> result(7, 0);

	double TP, TN, FP, FN;

	double whitePixelAccurIm = 0;
	double whitePixelAppIm = 0;
	FP = 0;
	FN = 0;
	TP = 0;
	TN = 0;
	for (int i = 0; i < accurImage.rows; ++i)
	{
		for (int j = 0; j < accurImage.cols; ++j)
		{
			if ((accurImage.at<cv::Vec3b>(i, j)[2] == 255) && ((accurImage.at<cv::Vec3b>(i, j)[1] == 255)) && ((accurImage.at<cv::Vec3b>(i, j)[0] == 255)))
			{
				whitePixelAccurIm++;
				if ((approxImage.at<cv::Vec3b>(i, j)[2] == 255) && ((approxImage.at<cv::Vec3b>(i, j)[1] == 255)) && ((approxImage.at<cv::Vec3b>(i, j)[0] == 255)))
				{
					TP++;
				}
			}
			if ((accurImage.at<cv::Vec3b>(i, j)[2] == 0) && ((accurImage.at<cv::Vec3b>(i, j)[1] == 0)) && ((accurImage.at<cv::Vec3b>(i, j)[0] == 0)))
			{
				if ((approxImage.at<cv::Vec3b>(i, j)[2] == 0) && ((approxImage.at<cv::Vec3b>(i, j)[1] == 0)) && ((approxImage.at<cv::Vec3b>(i, j)[0] == 0)))
				{
					TN++;
				}
			}
			if ((approxImage.at<cv::Vec3b>(i, j)[2] == 255) && ((approxImage.at<cv::Vec3b>(i, j)[1] == 255)) && ((approxImage.at<cv::Vec3b>(i, j)[0] == 255)))
			{
				whitePixelAppIm++;
			}
		}
	}
	FN = whitePixelAccurIm - TP;
	FP = whitePixelAppIm - TP;

	if ((TP + FP) != 0)
	{
		result[1] = TP / (TP + FP);
	}
	if ((TP + FN) != 0)
	{
		result[2] = TP / (TP + FN);
		result[5] = FN / (TP + FN);
	}
	if ((TN + FP) != 0)
	{
		result[3] = TN / (TN + FP);
		result[4] = FP / (TN + FP);
	}
	if ((result[2] != 0) || (result[1] != 0))
	{
		result[0] = 2 * result[1] * result[2] / (result[1] + result[2]);
	}
	if ((TP + FN + FP + TN) != 0)
	{
		result[6] = 100 * (FN + FP) / (TP + FN + FP + TN);
	}

	return result;
}

void metricCalculation(const std::string& videoObject, const std::string& videoDetectedObject, const std::string& path)
{
	cv::VideoCapture capture(videoObject);
	cv::VideoCapture captureObject(videoDetectedObject);

	if ((!capture.isOpened()) || (!captureObject.isOpened()))
	{
		std::cerr << "Can not open the video!" << std::endl;
		exit(-1);
	}

	cv::Mat frame;
	cv::Mat frameObject;

	int i = 0;
	while (true)
	{
		i++;
		capture >> frame;
		captureObject >> frameObject;

		if ((frame.empty()) || (frameObject.empty()))
		{
			break;
		}

		std::vector<double> eval = F_Measure(frame, frameObject);
		wrFile(eval, std::string(path + "measures.txt"), std::ios::app, std::to_string(i));
	}
	captureObject.release();
	capture.release();
}

void ViBeDemo();

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{ help h                |								| }"
		                                     "{ @input_video          | ./Video.avi | }"
		                                     "{ @input_object_video   | ./Object.avi | }"
	                                         "{ @output_path          | ./ | }");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const auto& fileName = parser.get<std::string>("@input_video");
	const auto& fileNameObject = parser.get<std::string>("@input_object_video");
	const auto& outputPath = parser.get<std::string>("@output_path");

	const auto nameVideoDet = std::string(outputPath + "detectedObject.avi");

    std::cout << "Select the algorithm: \n"
        << "Diff  - Basic difference \n"
        << "BU    - Basic difference with background updating \n"
        << "GMM   - Gaussian mixture model algorithm \n"
        << "MM    - MinMax algoruthm \n"
        << "1G    - One Gaussian \n"
        << "VB    - ViBe algorithm \n";

    std::string algorithmName;
    std::cin >> algorithmName;

    std::unique_ptr<SegmentMotionBase> ptr(SegmentMotionBase::CreateAlgorithm(algorithmName));

    if (ptr)
    {
        ptr->Run(fileName, nameVideoDet);
    }
    else
    {
        std::cout << "Run ViBe by default" << std::endl;
        ViBeDemo();
    }

	metricCalculation(fileNameObject, nameVideoDet, outputPath);

    return 0;
}
