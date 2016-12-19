///@File: ObjectTrackingLK.cpp
///@Brief: implementation of ObjectTrackingLK class
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "ObjectTrackingLK.h"

cv::Point2f point;
bool addRemovePt = false;

void ObjectTrackingLK::help()
{
    std::cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo()\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - auto-initialize tracking\n"
        "\tc - delete all the points\n"
        "\tn - switch the \"night\" mode on/off\n"
        "To add/remove a feature point click it\n" << std::endl;
}

void drawRectangles(cv::Mat& image, const std::vector<cv::Point2f>& points, const std::vector<int>& timeStopRect, const int& timeStop, const int& size_rect)
{
	std::list<cv::Rect2i> rectangles;

	for (size_t i = 0; i < timeStopRect.size(); ++i)
	{
		if (timeStopRect[i] > timeStop)
		{
			rectangles.push_back(cv::Rect2i(cv::Point2i(points[i].x - size_rect / 2, points[i].y - size_rect / 2), cv::Size(size_rect, size_rect)));
		}
	}

	double w, h, dx, dy;
	cv::Rect2i current_rect, rect_j;

	for (auto it1 = rectangles.begin(); it1 != rectangles.end(); ++it1)
	{
		current_rect = *it1;

		for (auto it2 = rectangles.begin(); it2 != rectangles.end(); ++it2)
		{
			rect_j = *it2;
			if ( current_rect == rect_j )
				continue;

			w = (rect_j.width + current_rect.width) / 2;
			h = (rect_j.height + current_rect.height) / 2;
			dx = abs(rect_j.x - current_rect.x + (rect_j.width - current_rect.width) / 2);
			dy = abs(rect_j.y - current_rect.y + (rect_j.height - current_rect.height) / 2);

			if ((dx <= w) && (dy <= h))
			{
				current_rect = cv::Rect2i(cv::Point2i(cv::min(rect_j.x, current_rect.x), cv::min(rect_j.y, current_rect.y)),
				cv::Size(cv::max(rect_j.x + rect_j.width, current_rect.x + current_rect.width) - cv::min(rect_j.x, current_rect.x),
				cv::max(rect_j.y + rect_j.height, current_rect.y + current_rect.height) - cv::min(rect_j.y, current_rect.y)));

				*it1 = current_rect;
				rectangles.erase(it2);
				it2 = rectangles.begin();
			}
		}
	}

	for (const auto& val : rectangles)
	{
		cv::rectangle(image, val, cv::Scalar(0, 0, 255), 2, 8);
	}

	rectangles.clear();
}

void ObjectTrackingLK::Run(cv::VideoCapture& capture, cv::VideoWriter& videoOut, const std::string& name_background)
{
    help();

	int size_rect = 50;
	double min_dist = 0.01;
	int timeStop = 6;

    const int MAX_COUNT = 500;
    bool needToInit = true;
    bool nightMode = false;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10), winSize(31, 31);

    cv::namedWindow(GetName(), 1);
    cv::setMouseCallback(GetName(), onMouse, 0);

    cv::Mat gray, prevGray, image, frame, foreground, background;
    std::vector<cv::Point2f> points[2];
	std::vector<int> timeStopRect, new_timeStopRect;

    int win_Size = 3;
    cv::createTrackbar("WinSize", GetName(), &win_Size, 10);
	cv::createTrackbar("Sizes rectangles", GetName(), &size_rect, 100);
	cv::createTrackbar("Stop time", GetName(), &timeStop, 30);

	background = cv::imread(name_background, CV_LOAD_IMAGE_GRAYSCALE);

    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;

        frame.copyTo(image);
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        if (nightMode)
            image = cv::Scalar::all(0);

        if (needToInit)
        {
            // automatic initialization

			cv::absdiff(gray, background, foreground);

			cv::threshold(foreground, foreground, 17, 255, cv::THRESH_BINARY);

            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, foreground, win_Size, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
            addRemovePt = false;
        }
        else if (!points[0].empty())
        {
            std::vector<uchar> status;
            std::vector<float> err;

			if (timeStopRect.empty())
				timeStopRect.resize(points[0].size(), 0);

            if (prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                5, termcrit, 0, 0.001);
            size_t i, k;
            for (i = k = 0; i < points[1].size(); i++)
            {
                if (addRemovePt)
                {
                    if (cv::norm(point - points[1][i]) <= 5)
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if (!status[i])
                    continue;

				if (cv::norm(points[1][i] - points[0][i]) < min_dist)
				{
					new_timeStopRect.push_back(timeStopRect[i] + 1);
				}
				else
				{
					new_timeStopRect.push_back(0);
				}

                points[1][k++] = points[1][i];
                circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
            }
			points[1].resize(k);

			timeStopRect = new_timeStopRect;
			new_timeStopRect.clear();
        }

        if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
        {
            std::vector<cv::Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix(gray, tmp, winSize, cv::Size(-1, -1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
		drawRectangles(image, points[1], timeStopRect, timeStop, size_rect);
        imshow(GetName(), image);

		videoOut.write(image);

        char c = (char)cv::waitKey(10);
        if (c == 27)
            return;
        switch (c)
        {
        case 'r':
            needToInit = true;
			points[0].clear();
			points[1].clear();
			timeStopRect.clear();
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
			timeStopRect.clear();
            break;
        case 'n':
            nightMode = !nightMode;
			points[0].clear();
			points[1].clear();
			timeStopRect.clear();
            break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
}

void ObjectTrackingLK::onMouse(int event, int x, int y, int, void *)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        point = cv::Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}
