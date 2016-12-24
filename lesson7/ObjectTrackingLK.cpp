///@File: ObjectTrackingLK.cpp
///@Brief: implementation of ObjectTrackingLK class
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "ObjectTrackingLK.h"

cv::Point2f point;
bool addRemovePt = false;


void ObjectTrackingLK::Run(cv::VideoCapture & capture)
{
	const int MAX_COUNT = 500;
	int win_Size = 12;
	bool needToInit = true;
    bool nightMode = false;

	cv::Mat gray, prevGray, image, frame, foreground, background;

	std::vector<cv::Point2f> points[2];
	std::vector<cv::Rect> static_objects, static_objects_merged;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10), winSize(31, 31);


	background = cv::imread("lab7_background.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (background.empty())
	{
		std::cerr << "error loading background image" << std::endl;
		return;
	}

	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
	cv::Size S = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps = capture.get(CV_CAP_PROP_FPS);
	cv::VideoWriter outputVideo("lab7_processed.avi", ex, fps, S);
	if (!outputVideo.isOpened())
	{
		std::cerr << "error opening output video for write" << std::endl;
		return;
	}

    cv::namedWindow(GetName(), 1);
    
    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;

        frame.copyTo(image);
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		cv::absdiff(gray, background, foreground);
		cv::threshold(foreground, foreground, 50, 255, CV_THRESH_BINARY);

		size_t min_i = 1e9, max_i = 0, min_j = 1e9, max_j = 0;
		for (size_t i = 0; i < foreground.rows; i++)
		{
			for (size_t j = 0; j < foreground.cols; j++)
			{
				if (foreground.at<uint8_t>(i, j) == 255)
				{
					if (i > max_i)
						max_i = i;
					if (i < min_i)
						min_i = i;
					if (j > max_j)
						max_j = j;
					if (j < min_j)
						min_j = j;
				}
			}
		}
		cv::Rect object_rect(cv::Point(min_i, min_j), cv::Point(max_i+1, max_j+1));
		foreground(object_rect) = 255*cv::Mat::ones(object_rect.size(), CV_8U);

        if (nightMode)
            image = cv::Scalar::all(0);

        if (needToInit)
        {
            // automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, foreground, win_Size, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
            addRemovePt = false;
        }
        else if (!points[0].empty())
        {
            std::vector<uchar> status;
            std::vector<float> err;
            if (prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 5, termcrit, 0, 0.001);
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

                points[1][k++] = points[1][i];
                circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
            }
            points[1].resize(k);

			// create rectangles around static objects
			static_objects.clear();
			for (size_t i = 0; i < points[1].size(); i++)
			{
				float dx = points[1].at(i).x - points[0].at(i).x;
				float dy = points[1].at(i).y - points[0].at(i).y;
				if (dx + dy < 0.5)
				{
					cv::Rect highlight(cv::Point(points[1].at(i)) - cv::Point(25, 25), cv::Size(50, 50));
					static_objects.push_back(highlight);
				}
			}

			// merge and draw rectangles
			static_objects_merged.clear();
			cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
			cv::Size scaleFactor(10, 10);
			for (int i = 0; i < static_objects.size(); i++)
			{
				cv::Rect box = static_objects.at(i) + scaleFactor;
				cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED);
			}
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			for (int j = 0; j < contours.size(); j++)
				static_objects_merged.push_back(cv::boundingRect(contours.at(j)));
			for (size_t i = 0; i < static_objects_merged.size(); i++)
				cv::rectangle(image, static_objects_merged.at(i), cv::Scalar(0, 0, 255));
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
        imshow(GetName(), image);
		outputVideo << image;

        char c = (char)cv::waitKey(10);
        if (c == 27)
            return;
        switch (c)
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
}