///@File: lesson5.cpp
///@Brief: main function
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"

#pragma comment(lib, "opencv_world300d.lib")

int main(int argc, char** argv)
{
	std::string filename = "movie.avi";

	cv::VideoCapture capture(filename);
	if (!capture.isOpened())
	{
		std::cerr << "error opening file " << filename << std::endl;
		return -1;
	}

	std::unique_ptr<IObjectTracking> ptr(IObjectTracking::CreateAlgorythm("LK"));
	ptr->Run(capture);

	return 0;
}
