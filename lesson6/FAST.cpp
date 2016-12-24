#include "stdafx.h"
#include "FAST.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#pragma comment(lib, "opencv_world300d.lib")

#include <vector>

#define DARKER 1
#define SIMILAR 2
#define BRIGHTER 3

FastDetector::FastDetector() {}

CircleData FastDetector::exploreCircle(const cv::Mat& image,
                                       const int x, const int y, const int r)
{
	int centerPix = image.data[x + y*image.cols];
    std::vector<cv::Point> circ = getCircle(x, y, r);
    int szCirc = (int)circ.size();

    int dkconsec=0, brconsec=0, consec=0;
    int angle_dkb=-1, angle_dke=-1;
    int angle_brb=-1, angle_bre=-1;
    int angle=-1;

    unsigned int intensity=0;

    for(int i=0; i<2*szCirc; i++) {
        int cx = circ[i%szCirc].x;
        int cy = circ[i%szCirc].y;
		int piPix = image.data[cx + cy*image.cols];
        if (centerPix > piPix + THRESHOLD) {
            if (i==0) {
                angle=getAngle(i%szCirc,r);
            } else if (intensity == DARKER) {
                consec++;
            } else {
                if (intensity == BRIGHTER && consec > brconsec) {
                    angle_bre = getAngle(i%szCirc,r);
                    angle_brb = angle;
                    brconsec = consec;
                }
                angle=getAngle(i%szCirc,r);
                consec=1;
            }
            intensity = DARKER;
        } else if (centerPix < piPix - THRESHOLD) {
            if (i==0) {
                angle=getAngle(i%szCirc,r);
            } else if (intensity == BRIGHTER) {
                consec++;
            } else {
                if (intensity == DARKER && consec > dkconsec) {
                    angle_dke = getAngle(i%szCirc,r);
                    angle_dkb = angle;
                    dkconsec = consec;
                }
                angle=getAngle(i%szCirc,r);
                consec=1;
            }
            intensity = BRIGHTER;
        } else {
            if (intensity == BRIGHTER) {
                if (consec > brconsec) {
                    angle_bre = getAngle(i%szCirc,r);
                    angle_brb = angle;
                    brconsec = consec;
                }
            } else if (intensity == DARKER) {
                if (consec > dkconsec) {
                    angle_dke = getAngle(i%szCirc,r);
                    angle_dkb = angle;
                    dkconsec = consec;
                }
            }
            intensity = SIMILAR;
        }
    }
    
    CircleData cd(angle_dkb, angle_dke, angle_brb, angle_bre, dkconsec, brconsec);
    return cd;
}

int FastDetector::getOrientation(const CircleData& cd)
{
    int orientation = -1;
    if (cd.dkConsec > cd.brConsec) {
        orientation = (cd.dkBeginAngle + cd.dkEndAngle) / 2;
    } else {
        orientation = (cd.brBeginAngle + cd.brEndAngle) / 2;
    }
    return orientation;
}

bool FastDetector::crossCheck(const cv::Mat& image, const int x, const int y)
{
	int centerPix = image.data[x + y*image.cols];

	int p1Pix = image.data[x + (y - 3) * image.cols];
	int p5Pix = image.data[(x + 3) + y * image.cols];
	int p9Pix = image.data[x + (y + 3) * image.cols];
	int p13Pix = image.data[(x - 3) + y * image.cols];

    int dkconsec=0, brconsec=0;
    if (centerPix > p1Pix + THRESHOLD) dkconsec++; // top point
    else if (centerPix < p1Pix - THRESHOLD) brconsec++;

    if (centerPix > p5Pix + THRESHOLD) dkconsec++; // right point
    else if (centerPix < p5Pix - THRESHOLD) brconsec++;

    if (centerPix > p9Pix + THRESHOLD) dkconsec++; // bottom point
    else if (centerPix < p9Pix - THRESHOLD) brconsec++;

    if (centerPix > p13Pix + THRESHOLD) dkconsec++; // left point
    else if (centerPix < p13Pix - THRESHOLD) brconsec++;

    if (dkconsec==3 || brconsec==3) return true;
    else return false;
}

bool FastDetector::defaultCheck(const cv::Mat& image, const int x, const int y)
{
    CircleData cd = exploreCircle(image, x, y, 3);
    int dkconsec = cd.dkConsec;
    int brconsec = cd.brConsec;
    if (dkconsec>11 || brconsec>11)
	{
        return true;
    }
    return false;
}

bool FastDetector::extendedCheck(const cv::Mat& image, const int x, const int y)
{
    CircleData cd2 = exploreCircle(image, x, y, 2);
    CircleData cd3 = exploreCircle(image, x, y, 3);
    CircleData cd4 = exploreCircle(image, x, y, 4);

    int orientation2 = getOrientation(cd2);
    int orientation3 = getOrientation(cd3);
    int orientation4 = getOrientation(cd4);

    int theta23 = abs(orientation2-orientation3);
    int theta34 = abs(orientation3-orientation4);

    if (theta23 < ANGULAR_THRESHOLD && theta34 < ANGULAR_THRESHOLD)
	{
        return true;
    } 
	else 
	{
        return false;
    }
}

void FastDetector::detect(const cv::Mat& image, const int i_THRESHOLD)
{
	THRESHOLD = i_THRESHOLD;
    cv::Mat image_gray;
	cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    for(int y=5; y<image_gray.rows-5; y++)
	{
        for(int x=5; x<image_gray.cols-5; x++)
		{
            if (crossCheck(image_gray,x,y))
			{
                if (defaultCheck(image_gray,x,y))
				{
                    if (extendedCheck(image_gray,x,y))
					{
                        //detectedPoints.push_back(cv::Point(x,y));
						detectedPoints.push_back(cv::KeyPoint(cv::Point(x, y), 2));
                    }
                }
            }
        }
    }
}

void FastDetector::express(cv::Mat& image)
{
    int sz_detectedPoints = (int)detectedPoints.size();
    for(int i=0; i < sz_detectedPoints; i++) 
	{
        //circle(image, detectedPoints[i], 3, cv::Scalar(0,0,255), 1, 8, 0);
		circle(image, detectedPoints[i].pt, 3, cv::Scalar(0, 0, 255), 1, 8, 0);
    }
}

std::vector<cv::KeyPoint> FastDetector::getFeaturePoints()
{
	return detectedPoints;
}
