#include "SegmentMotionBase.h"

#include <iostream>

#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include "SegmentMotion1G.h"

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionBase::Run(const std::string &video_file)
{
    cv::VideoCapture capture(video_file);

	cv::VideoWriter outputVideo;


	try {
		outputVideo.open(std::string("test.avi"), CV_FOURCC('M','J','P','G'), 20.0, {512, 511});
	} catch(...) {
		std::cerr << "Incorrect parameters for frameWriter." << std::endl;
		exit(-1);
	}


    if (!capture.isOpened())
    {
        std::cerr << "Can not open the camera !" << std::endl;
        exit(-1);
    }

    createGUI();

	cv::Mat frame;

    while (true)
    {
		capture >> frame;

		if (frame.empty())
			break;

        m_foreground = process(frame);
        cv::imshow(GetName(), m_foreground);

		cv::Mat buf(frame.rows, frame.cols, CV_8UC3);

		cv::cvtColor(m_foreground, buf, cv::COLOR_GRAY2BGR);

		outputVideo.write(buf);

        if (cv::waitKey(1) >= 0)
        {
            break;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
SegmentMotionBase* SegmentMotionBase::CreateAlgorithm(std::string& algorithmName)
{
	if (algorithmName == "1G")
	{
		return new SegmentMotion1G();
	}
    else
    {
        return nullptr;
    }
}
