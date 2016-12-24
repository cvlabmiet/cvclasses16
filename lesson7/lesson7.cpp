///@File: lesson5.cpp
///@Brief: main function
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{ help h                |                                           | }"
		                                     "{ @input_video          | ./Surveillance System/ImageWithObject.avi | }"
		                                     "{ @background           | ./Surveillance System/Rostislav.png       | }"
		                                     "{ @output_path          | ./Surveillance System/                    | }");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const auto& fileName = parser.get<std::string>("@input_video");
	const auto& outputPath = parser.get<std::string>("@output_path");
	const auto& name_background = parser.get<std::string>("@background");

	const auto& nameVideoDet = std::string(outputPath + "detectedObject.avi");

    std::unique_ptr<IObjectTracking> ptr(IObjectTracking::CreateAlgorythm("LK"));
    if (ptr)
    {
        cv::VideoCapture capture;

		cv::VideoWriter  videoOut;
		
		capture = cv::VideoCapture(fileName);

		int codec = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
		double fps = capture.get(CV_CAP_PROP_FPS);

		videoOut.open(nameVideoDet, codec, fps, cv::Size((int)capture.get(CV_CAP_PROP_FRAME_HEIGHT), (int)capture.get(CV_CAP_PROP_FRAME_WIDTH) ));

		if (!videoOut.isOpened())
		{
			std::cerr << "Could not open the output video file for write\n";
			exit(1);
		}

        if (!capture.isOpened())
        {
            std::cout << "Capture opening failed.\n";
            exit(1);
        }

        ptr->Run(capture, videoOut, name_background);

		videoOut.release();
		capture.release();
    }
    else
    {
        std::cerr << "Invalid name of algorythm." << std::endl;
    }

    return 0;
}
