#include "fakevideocapture.h"

#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>

void draw_shape(cv::Mat& image, Shape shape, cv::Point2i center, cv::Scalar color)
{
    const int radius = 50;
    switch (shape) {
    case Shape::romb : {
        cv::Point2i pts[4];
        pts[0] = cv::Point2i(center.x - radius / 2, center.y);
        pts[1] = cv::Point2i(center.x, center.y + radius / 2);
        pts[2] = cv::Point2i(center.x + radius / 2, center.y);
        pts[3] = cv::Point2i(center.x, center.y - radius / 2);
        cv::fillConvexPoly(image, pts, 4, color);
        break;
    }
    case Shape::circle : {
        cv::circle(image, center, radius, color, -1);
    }
    }
}

cv::Mat FakeVideoCapture::next_frame()
{
    const double timeToAppear = 10;
    const double travelTime = 20;

    cv::Mat frame;
    foreground.copyTo(frame);

    motionBinary = cv::Mat::zeros(frame.size(), CV_8U);

    double time = frameN / (double) fps;
    if (time < timeToAppear) {
        // do nothing
    }
    else if (time < timeToAppear + travelTime) {
        // draw shape
        const int height = (foreground.size()).height;
        const int width = (foreground.size()).width;

        const double amp = height / 5;
        const double freq = 2 * 3.14 / (width / 5);
        const double constant = height / 2;

        int centerX = static_cast<int>(width * (time - timeToAppear) / travelTime);
        int centerY = static_cast<int>(constant + amp * sin(freq * centerX));

        draw_shape(frame, shape, cv::Point2i(centerX, centerY), color);
        draw_shape(motionBinary, shape, cv::Point2i(centerX, centerY), cv::Scalar(255));
    }

    frameN++;
    return frame;
}

void FakeVideoCapture::calc_metrics(const cv::Mat& detBinary)
{
    cv::Mat truePos, trueNeg, falsePos, falseNeg, tmp, tmp2;

    cv::bitwise_and(detBinary, motionBinary, truePos);
    const int TP = cv::countNonZero(truePos);

    cv::bitwise_not(motionBinary, tmp);
    cv::bitwise_and(detBinary, tmp, falsePos);
    const int FP = cv::countNonZero(falsePos);

    cv::bitwise_not(detBinary, tmp);
    cv::bitwise_and(motionBinary, tmp, falseNeg);
    const int FN = cv::countNonZero(falseNeg);

    cv::bitwise_not(detBinary, tmp);
    cv::bitwise_not(motionBinary, tmp2);
    cv::bitwise_and(tmp, tmp2, trueNeg);
    const int TN = cv::countNonZero(trueNeg);

    double recall      = TP / (double) (TP + FN);
    double precision   = TP / (double) (TP + FP);
    double specificity = TN / (double) (TN + FP);
    double fpr         = FP / (double) (TN + FP);
    double fnr         = FN / (double) (TP + FN);
    double pwc         = 100 * (FN + FP) / (double) (TP + FN + FP + TN);

    if (TP == 0) {
        if (FN == 0) {
            // 0 out of 0 true points discovered
            recall = 1;
            fnr = 0;
        }
        if (FP == 0) {
            // 0 points selected, all of them are relevant
            precision = 1;
        }
    }
    double f_measure   = 2 * precision * recall / (precision + recall);

    fwrite(&recall, sizeof(double), 1, fp);
    fwrite(&precision, sizeof(double), 1, fp);
    fwrite(&specificity, sizeof(double), 1, fp);
    fwrite(&fpr, sizeof(double), 1, fp);
    fwrite(&fnr, sizeof(double), 1, fp);
    fwrite(&pwc, sizeof(double), 1, fp);
    fwrite(&f_measure, sizeof(double), 1, fp);
}
