///@File: SegmentMotion1G.cpp
///@Brief: Contains implementation of segmentation based on One Gaussian
///@Author: Kuksova Svetlana
///@Date: 26 October 2015

#include "SegmentMotion1G.h"

#include <iostream>
#include <iterator>

#include "opencv2\video\video.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotion1G::process(cv::Mat& currentFrame)
{
	if (!m_algorithmPtr)
	{
		m_algorithmPtr = cv::createBackgroundSubtractorMOG2(m_params.historySize, m_params.T, false);
		(*m_algorithmPtr).setNMixtures(1);
		
	}

	cv::Mat result;

	m_algorithmPtr->apply(currentFrame, result);

	return result;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotion1G::createGUI()
{
    const std::string windowName = GetName();
    cv::namedWindow(windowName);

	m_params.historySize = 3*24;
    m_params.T = 10;

    cv::createTrackbar("History", windowName, reinterpret_cast<int*>(&m_params.historySize), 1000);
    cv::createTrackbar("T", windowName, &m_params.T, 255);
}
