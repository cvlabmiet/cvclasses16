#ifndef _FAST_H_
#define _FAST_H_

#include <opencv2/core/core.hpp>

#include "helper.h"

//#define THRESHOLD 30 // smallest difference in pixel intensity noticed as possible corner
#define ANGULAR_THRESHOLD 10 // maximum allowable difference in orientation of corner

class FastDetector {
    //std::vector<cv::Point> detectedPoints;
	std::vector<cv::KeyPoint> detectedPoints;
    CircleData exploreCircle(const cv::Mat& image, const int x, const int y, const int r);
    int getOrientation(const CircleData& cd);
    bool crossCheck(const cv::Mat& image, const int x, const int y);
    bool defaultCheck(const cv::Mat& image, const int x, const int y);
    bool extendedCheck(const cv::Mat& image, const int x, const int y);
	int THRESHOLD;
  public:
    FastDetector();
	~FastDetector() {};
    void detect(const cv::Mat& image, const int i_THRESHOLD);
	std::vector<cv::KeyPoint> getFeaturePoints();
    void express(cv::Mat& image);
};

#endif
