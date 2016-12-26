///@File: ObjectTrackingLK.cpp
///@Brief: implementation of ObjectTrackingLK class
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "ObjectTrackingLK.h"

#include <iostream>
#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>

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

void ObjectTrackingLK::Run(cv::VideoCapture & capture)
{
    help();

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10), winSize(31, 31);

    cv::namedWindow(GetName(), 1);
    cv::setMouseCallback(GetName(), onMouse, 0);

    cv::Mat gray, prevGray, image, frame;
    std::vector<cv::Point2f> points[2];

    int win_Size = 3;
    cv::createTrackbar("WinSize", GetName(), &win_Size, 10);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    std::vector<bool> movedPts;
    std::vector<bool> prevMovedPts;

    bool firstIter = true;
    bool stopDetected = false;
    for (;;)
    {
        capture >> frame;
        if (frame.empty()) {
            std::cout << "Frame is empty" << std::endl;
            break;
        }

        frame.copyTo(image);
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        if (nightMode)
            image = cv::Scalar::all(0);

        if (needToInit)
        {
            // automatic initialization
//            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, cv::Mat(), win_Size, 0, 0.04);
//            cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
            std::vector<cv::KeyPoint> kPoints;
            detector->detect(gray, kPoints);

            auto kPtComp = [] (const cv::KeyPoint& pt1, const cv::KeyPoint& pt2) -> bool {
                return pt1.response < pt2.response;
            };
            std::sort(kPoints.begin(), kPoints.end(), kPtComp);

            points[1].clear();
            const int ptNum = std::min(MAX_COUNT, (int)kPoints.size());
            for (int i = 0; i < ptNum; i++) {
                points[1].push_back(kPoints[i].pt);
            }

            addRemovePt = false;
        }
        else if (!points[0].empty())
        {
            std::vector<uchar> status;
            std::vector<float> err;
            if (prevGray.empty())
                gray.copyTo(prevGray);
            cv::calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                5, termcrit, 0, 0.001);
            movedPts.clear();
            for (int i = 0; i < err.size(); i++) {
                if (status[i]) {
                    if (err[i] > 0.5) {
                        // point has moved
                        movedPts.push_back(true);
                        continue;
                    }
                }
                movedPts.push_back(false);
            }
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
                if (movedPts[i]) {
                    circle(image, points[1][i], 10, cv::Scalar(0, 0, 255), -1, 8);
                }
                else {
                    circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
                }
            }
            points[1].resize(k);
        }

        if (firstIter) {
            prevMovedPts = std::vector<bool>(movedPts.size(), false);
            firstIter = false;
        }
        // detect stop of motion
        std::vector<bool> changedMovement;
        for (int i = 0; i < movedPts.size(); i++) {
            if ((prevMovedPts[i] == true) && (movedPts[i] == false)) {
                changedMovement.push_back(true);
            }
            else {
                changedMovement.push_back(false);
            }
        }
        int numStopped = std::count(changedMovement.begin(), changedMovement.end(), true);
        int numMoved = std::count(prevMovedPts.begin(), prevMovedPts.end(), true);
        std::cout << "Num moved: " << numMoved << ", num stopped: " << numStopped << std::endl;
        if ((numStopped >= 0.9 * numMoved) && (numMoved > 10)) {
            // stop detected
            std::cout << "Stop detected" << std::endl;
            std::vector<cv::Point2f> pts;
            for (int i = 0; i < points[1].size(); i++) {
                if (changedMovement[i]) {
                    pts.push_back(points[1][i]);
                }
            }
            std::cout << pts.size() << std::endl;
            cv::Rect bndRect = cv::boundingRect(pts);
            std::cout << bndRect.x << " " << bndRect.y << std::endl;
            std::cout << bndRect.width << " " << bndRect.height << std::endl;
            cv::rectangle(image, bndRect, cv::Scalar(0, 0, 255), 4);
            stopDetected = true;
        }

        if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
        {
            std::vector<cv::Point2f> tmp;
            tmp.push_back(point);
//            cornerSubPix(gray, tmp, winSize, cv::Size(-1, -1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
        imshow(GetName(), image);

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
        std::swap(prevMovedPts, movedPts);
        cv::swap(prevGray, gray);
        if (stopDetected) {
            cv::waitKey(0);
            stopDetected = false;
        }
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
