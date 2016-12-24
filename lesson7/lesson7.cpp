///@File: lesson5.cpp
///@Brief: main function
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"

int main(int argc, char** argv)
{
	std::string filename = "lab7_source.avi";

	cv::VideoCapture capture(filename);
	if (!capture.isOpened())
	{
		std::cerr << "error opening file " << filename << std::endl;
		return -1;
	}

	std::unique_ptr<IObjectTracking> ptr(IObjectTracking::CreateAlgorythm("LK"));
	ptr->Run(capture);

	return 0;

	/*
    std::cout << "Choose algorythm or press q to quit:\n" <<
        "LK - Lucas-Kanade algorythm\n" <<
        "TK - Tomasi-Kanade algorythm\n" <<
        "STK - Shi-Tomasi-Kanade algorythm\n";

    std::string name;
    std::cin >> name;

    if (name == "q")
        return 0;

    std::unique_ptr<IObjectTracking> ptr(IObjectTracking::CreateAlgorythm(name));
    if (ptr)
    {
        cv::VideoCapture capture;
        if (argc == 1)
            capture = cv::VideoCapture(0);
        else
            capture = cv::VideoCapture(argv[1]);

        if (!capture.isOpened())
        {
            std::cout << "Capture opening failed.\n";
            exit(1);
        }

        ptr->Run(capture);
    }
    else
    {
        std::cerr << "Invalid name of algorythm." << std::endl;
    }

    return 0;
	*/
}
