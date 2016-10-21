///@File: main.cpp
///@Brief: Contains implementation of entry point of the application
///@Author: Roman Golovanov
///@Date: 08 September 2015

#include "stdafx.h"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <map>
#include <functional>

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <fstream>

const std::string c_origWindowName = "Original Picture";
const std::string c_edgeWindowName = "Edge Picture";

int g_threshold = 0;
int g_operatorId = 5;
std::pair<int, int> g_lastThOp = { -1, -1 };

///@brief Edge operator specification
struct EdgeOperator
{
   std::string Name;
   float fx[3*3]; // kernel for horizontal derivation
   float fy[3*3]; // kernel for vertical derivation
};

const std::string LoGName = "LoG";
const std::string CannyName = "Canny";

cv::Mat imWithEdjes;

///@brief array of edge operators
std::vector<EdgeOperator> g_operatorSpec = {
   EdgeOperator{ "Robert\'s",       { -1,  1, 0 , 0, 0, 0, 0, 0, 0 },
                                    { -1,  0, 0,  1, 0, 0, 0, 0, 0 } },
   EdgeOperator{ "Robert\'s Cross", { -1,  0, 0 , 0, 1, 0, 0, 0, 0 },
                                    {  0, -1, 0,  1, 0, 0, 0, 0, 0 } },
   EdgeOperator{ "Prewitt",         { -1,  0,  1, -1, 0, 1, -1, 0, 1 },
                                    { -1, -1, -1,  0, 0, 0,  1, 1, 1 } },
   EdgeOperator{ "Sobel",           { -1,  0,  1, -2, 0, 2, -1, 0, 1 },
                                    { -1, -2, -1,  0, 0, 0,  1, 2, 1 } },
   EdgeOperator{ "Scharr",          { -3,   0,  3, -10, 0, 10, -3,  0, 3 },
									{ -3, -10, -3,   0, 0,  0,  3, 10, 3 } },
   EdgeOperator{ "Diagonal Sobel",  { 0, 1, 2, -1, 0, 1, -2, -1, 0 },
									{ -2, -1, 0, -1, 0, 1, 0, 1, 2 }},
   EdgeOperator{ LoGName,           { 1,  1, 1 , 1, -8, 1, 1, 1, 1 },
                                    { 0,  0, 0,  0,  0, 0, 0, 0, 0 } },
   EdgeOperator{ CannyName },
};

/////////////////////////////////////////////////////////////////////
void apply1stDerivativeAlgo(cv::Mat& i_img, EdgeOperator& i_oper, cv::Mat& o_edges)
{
   cv::Mat imgX;
   cv::filter2D(i_img, imgX, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fx));

   cv::Mat imgY;
   cv::filter2D(i_img, imgY, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fy));

   cv::Mat imgMag;
   cv::magnitude(imgX, imgY, imgMag);

   double maxVal = 0;
   cv::minMaxLoc(imgMag, nullptr, &maxVal);

   cv::threshold(imgMag, o_edges, 0.01 * g_threshold * maxVal, 255, CV_THRESH_BINARY);
}

/////////////////////////////////////////////////////////////////////
void apply2ndDerivativeAlgo(cv::Mat& i_img, EdgeOperator& i_oper, cv::Mat& o_edges)
{
   cv::Mat log;
   cv::filter2D(i_img, log, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fx));

   double minVal = 0;
   double maxVal = 0;
   cv::minMaxLoc(log, &minVal, &maxVal);
   const double th = 0.01 * g_threshold * (std::max)(maxVal, std::abs(minVal));

   o_edges = cv::Mat::zeros(log.size(), log.type());

   for (int x = 1; x < log.cols - 1; ++x)
   {
      for (int y = 1; y < log.rows - 1; ++y)
      {
         const cv::Mat nghb = log.colRange({ x - 1, x + 2 }).rowRange({ y - 1, y + 2 });

         if (nghb.at<float>(0, 0) * nghb.at<float>(2, 2) < 0 && std::abs(nghb.at<float>(0, 0) - nghb.at<float>(2, 2)) > th ||
            nghb.at<float>(1, 0) * nghb.at<float>(1, 2) < 0 && std::abs(nghb.at<float>(1, 0) - nghb.at<float>(1, 2)) > th ||
            nghb.at<float>(2, 0) * nghb.at<float>(0, 2) < 0 && std::abs(nghb.at<float>(2, 0) - nghb.at<float>(0, 2)) > th ||
            nghb.at<float>(2, 1) * nghb.at<float>(0, 1) < 0 && std::abs(nghb.at<float>(2, 1) - nghb.at<float>(0, 1)) > th)
         {
            o_edges.at<float>(y, x) = 255;
         }
      }
   }
}

//output: [0] - F;[1] - Precision;[2] - Recall;[3] - TP;[4] - FP;[5] - FN;
std::vector<double> F_Measure(cv::Mat& accurImage, cv::Mat& approxImage)
{
	std::vector<double> result(6, 0);

	double whitePixelAccurIm = 0;
	double whitePixelAppIm = 0;


	for (int i = 0; i < accurImage.rows; ++i)
	{
		for (int j = 0; j < accurImage.cols; ++j)
		{
			if (accurImage.at<uchar>(i, j) == 255)
			{
				whitePixelAccurIm++;
				if (approxImage.at<float>(i, j) == 255)
				{
					result[3]++;
				}
			}
			if (approxImage.at<float>(i, j) == 255)
			{
				whitePixelAppIm++;
			}
		}
	}
	result[5] = whitePixelAccurIm - result[3];
	result[4] = whitePixelAppIm - result[3];

	if ((result[3] + result[4]) != 0)
	{
		result[1] = result[3] / (result[3] + result[4]);
		result[2] = result[3] / (result[3] + result[5]);
		result[0] = 2 * result[1] * result[2] / (result[1] + result[2]);
	}

	return result;
}

////////////////////////////////////////////////////////////////////
void writeImage(cv::Mat& i_image, const std::string& i_outputPath, const std::string& i_nameImage)
{
	std::string fName = i_outputPath + i_nameImage + ".png";

	try
	{
		cv::imwrite(fName, i_image);
	}
	catch (...)
	{
		std::cout << "Image can not be saved" << fName << std::endl;
	}
}

void wrFile(std::vector<double>& obj,const std::string& fileName, int mode, const std::string& mess)
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

///@brief provides frame processing
void processFrame(cv::Mat& i_image, const std::string& i_outputPath)
{
   if (g_lastThOp == std::make_pair(g_threshold, g_operatorId))
   {
      return; // nothing changed
   }

   g_lastThOp = { g_threshold, g_operatorId };

   auto& oper = g_operatorSpec[g_operatorId];

   cv::imshow(c_origWindowName, i_image);
   
   cv::Mat gray;
   cv::cvtColor(i_image, gray, CV_RGB2GRAY);

   cv::Mat edges;
   if (oper.Name == CannyName)
   {
      cv::Canny(gray, edges, g_threshold, g_threshold / 2);
   }
   else if(oper.Name == LoGName)
   {
      apply2ndDerivativeAlgo(gray, oper, edges);
   }
   else
   {
      apply1stDerivativeAlgo(gray, oper, edges);
   }

   const int bufSize = 100;
   char chint[bufSize];
   sprintf_s(chint, bufSize, "%s Operator %d", oper.Name.data(), g_threshold);
   std::string shint{ chint };

   std::vector<double> eval = F_Measure(imWithEdjes, edges);

   wrFile(eval, std::string(i_outputPath + "measures.txt"), std::ios::app, std::to_string(g_threshold));

   writeImage(edges, i_outputPath, shint);

   const auto txtSize = cv::getTextSize(shint, CV_FONT_HERSHEY_PLAIN, 1.0, 1, nullptr);
   cv::putText(edges, shint, { 10, 10 + txtSize.height }, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);

   cv::imshow(c_edgeWindowName, edges);
}

///@brief provides video capturing from the camera and frame processing
void processVideo(const std::string& i_outputPath)
{
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::cerr << "Can not open the camera !" << std::endl;
		return;
	}

	while (cv::waitKey(1) < 0)
	{
		cv::Mat frame;
		capture.read(frame);
		processFrame(frame, i_outputPath);
	}
}

///@brief provides image file processing
bool processImage(const std::string& i_name, const std::string& i_outputPath)
{

   cv::Mat img0 = cv::imread(i_name, 1);
   if (img0.empty())
   {
      std::cerr << "Couldn'g open image " << i_name << ". Usage: lesson3 <image_name>\n";
      return false;
   }

   while (cv::waitKey(1) < 0)
   {
      cv::Mat frame;
      processFrame(img0, i_outputPath);
   }

   return true;
}

void trackBarCallBack(int, void*)
{
   printf("Operator: %s; Threshold: %d\n", g_operatorSpec[g_operatorId].Name.data(), g_threshold);
}

///@brief Entry point
int _tmain(int argc, char** argv)
{
   cv::CommandLineParser parser(argc, argv, "{ help h           |                                    | }"
                                            "{ @input_image    | ./DSobelVsLoG/RostislavGray.png    | }"
											"{ @input_true_edge | ./DSobelVsLoG/RostislavGrayEdg.png | }"
	                                        "{ @output_path     | ./DSobelVsLoG/                    |  }");
   if (parser.has("help"))
   {
      parser.printMessage();
      return 0;
   }
   const auto& fileName = parser.get<std::string>("@input_image");
   const auto& nameImageWithEdjes = parser.get<std::string>("@input_true_edge");
   const auto& outputPath = parser.get<std::string>("@output_path");
   
   imWithEdjes = cv::imread(nameImageWithEdjes, 0);

   if (imWithEdjes.empty())
   {
	   std::cerr << "Couldn'g open image " << nameImageWithEdjes << std::endl;
   }

   /// Create window with original image
   cv::namedWindow(c_origWindowName, CV_WINDOW_AUTOSIZE);

   /// Create DEMO window
   cv::namedWindow(c_edgeWindowName, CV_WINDOW_AUTOSIZE);

   cv::createTrackbar("Threshold", c_edgeWindowName, &g_threshold, 100, trackBarCallBack);
   cv::createTrackbar("Operator",  c_edgeWindowName, &g_operatorId, g_operatorSpec.size() - 1, trackBarCallBack);

   if (!processImage(fileName, outputPath))
   {
      processVideo(outputPath);
   }

	return 0;
}
