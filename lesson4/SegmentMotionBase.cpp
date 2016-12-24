///@File: ISegmentMotion.cpp
///@Brief: Contains implementation of interface for SegmentMotion classes
///@Author: Vitaliy Baldeev
///@Date: 12 October 2015

#include "SegmentMotionBase.h"

#include <iostream>

#include "opencv2\video\video.hpp"
#include "opencv2\highgui\highgui.hpp"

#include "SegmentMotionDiff.h"
#include "SegmentMotionBU.h"
#include "SegmentMotionGMM.h"
#include "SegmentMotionMinMax.h"
#include "SegmentMotion1G.h"

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionBase::Run(const std::string& videoName, const std::string& videoDetectedObject)
{
    cv::VideoCapture capture(videoName);
	cv::VideoWriter  videoOut;

	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	double fps = capture.get(CV_CAP_PROP_FPS);

	videoOut.open(videoDetectedObject, codec, fps, {(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT), (int)capture.get(CV_CAP_PROP_FRAME_WIDTH)});

	if (!videoOut.isOpened())
	{
		std::cerr << "Could not open the output video file for write\n";
		exit(-1);
	}

    if (!capture.isOpened())
    {
        std::cerr << "Can not open the video!" << std::endl;
        exit(-1);
    }

    createGUI();

	cv::Mat frame;
	cv::Mat image_write(frame.rows, frame.cols, CV_8UC3);
    while (true)
    {
		capture >> frame;
		if (frame.empty())
		{
			break;
		}

		m_foreground = process(frame);
		cv::imshow(GetName(), m_foreground);

		cv::cvtColor(m_foreground, image_write, CV_GRAY2BGR);

		videoOut.write(image_write);
	
        if (cv::waitKey(1) >= 0)
        {
            break;
        }
    }

	videoOut.release();
	capture.release();
}

///////////////////////////////////////////////////////////////////////////////
SegmentMotionBase* SegmentMotionBase::CreateAlgorithm(std::string& algorithmName)
{
    if (algorithmName == "Diff")
    {
        return new SegmentMotionDiff();
    }
    else if (algorithmName == "BU")
    {
        return new SegmentMotionBU();
    }
    else if (algorithmName == "GMM")
    {
        return new SegmentMotionGMM();
    }
    else if (algorithmName == "MM")
    {
        return new SegmentMotionMinMax();
    }
	else if (algorithmName == "1G")
	{
		return new SegmentMotion1G();
	}
    else
    {
        return nullptr;
    }
}
