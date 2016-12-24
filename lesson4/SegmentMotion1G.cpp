#include "SegmentMotion1G.h"

#include <iostream>
#include <iterator>

#include "opencv2/video/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotion1G::process(cv::Mat &frame)
{
    if (m_params.historySize == 0)
    {
        m_params.historySize = 1;
    }

    if (m_frameBuffer.size() < m_params.historySize)
    {
		cv::Mat buf;
		cv::cvtColor(frame, buf, CV_BGR2GRAY);
        while (m_frameBuffer.size() < m_params.historySize)
        {
            m_frameBuffer.push_back(frame);
        }
    }
    if (m_frameBuffer.size() > m_params.historySize)
    {
        while (m_frameBuffer.size() > m_params.historySize)
        {
            m_frameBuffer.pop_front();
        }
    }

    cv::cvtColor(frame, frame, CV_BGR2GRAY);
    m_frameBuffer.pop_front();
    m_frameBuffer.push_back(frame);

    m_mMat = cv::Mat_<float>(frame.rows, frame.cols, 0.0);
    m_sigmaMat = cv::Mat_<float>(frame.rows, frame.cols, 0.0);


    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            std::list<cv::Mat>::const_iterator pos;

            for (pos = m_frameBuffer.begin(); pos != m_frameBuffer.end(); pos++)
            {
                const float val = static_cast<float>((*pos).at<uchar>(i, j));
                m_mMat(i, j) += val;
			}

			m_mMat(i, j) /= m_frameBuffer.size();

			for (pos = m_frameBuffer.begin(); pos != m_frameBuffer.end(); pos++)
            {
                const float val = static_cast<float>((*pos).at<uchar>(i, j));
                m_sigmaMat(i, j) += (val - m_mMat(i, j)) * (val - m_mMat(i, j));
			}
			m_sigmaMat(i, j) /= m_frameBuffer.size();
			m_sigmaMat(i, j) = std::sqrt(m_sigmaMat(i, j));
        }
    }

    // Detect foreground
    cv::Mat result(frame.rows, frame.cols, CV_8UC1);
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if ((abs(m_mMat(i, j) - static_cast<float>(frame.at<uchar>(i, j))) / m_sigmaMat(i, j)) <
                static_cast<float>(m_params.T))
            {
                result.at<uchar>(i, j) = static_cast<uchar>(255);
            }
            else
            {
				result.at<uchar>(i, j, 0) = static_cast<uchar>(0);
            }
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotion1G::createGUI()
{
    const std::string windowName = GetName();
    cv::namedWindow(windowName);

    m_params.historySize = 5;
    m_params.T = 2;

    cv::createTrackbar("History", windowName, reinterpret_cast<int*>(&m_params.historySize), 20);
    cv::createTrackbar("T", windowName, &m_params.T, 20);
}
