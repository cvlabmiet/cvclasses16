#ifndef FAKEVIDEOCAPTURE_H
#define FAKEVIDEOCAPTURE_H

#include <cstdio>
#include <iostream>

#include <opencv2/videoio/videoio.hpp>

enum class Shape {
    romb,
    circle
};

class FakeVideoCapture : public cv::VideoCapture
{
public:
    FakeVideoCapture(Shape shape, const cv::Mat& foreground, int fps, const char* filename = "metrics.bin") :
        shape(shape),
        foreground(foreground),
        fps(fps),
        frameN(0)
    {
        color = cv::mean(foreground);
        motionBinary = cv::Mat::zeros(foreground.size(), CV_8U);
        fp = fopen(filename, "wb");
        if (!fp) {
            std::cout << "Unable to open file " << filename << std::endl;
        }
    }

    virtual bool isOpened() const override
    {
        return true;
    }

    virtual FakeVideoCapture& operator >> (cv::Mat& image)
    {
        image = next_frame();
        return *this;
    }

    void calc_metrics(const cv::Mat& detBinary);

    ~FakeVideoCapture()
    {
        fclose(fp);
    }

private:
    Shape shape;
    cv::Mat foreground;
    cv::Scalar color;
    int fps;
    int frameN;
    FILE* fp;
    cv::Mat motionBinary;
    cv::Mat next_frame();
};

#endif // FAKEVIDEOCAPTURE_H
