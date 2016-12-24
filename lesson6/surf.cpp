#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

static const std::string refWinName = "Ref";
static const std::string tstWinName = "Tst";
static const std::string diffWinName = "Difference";

static const std::string minHessianTrackbarName = "min hessian value";
static const std::string nOctavesTrackbarName = "number of octaves";
static const std::string nOctLayersTrackbarName = "layers per octave";
static const std::string nMatchesTrackbarName = "number of matches";

static int minHessian;
static int nOctaves;
static int nOctLayers;
static int nMatches;

static const int maxMinHessian = 10000;
static const int maxNOctaves = 8;
static const int maxNOctLayers = 6;
static const int maxNMatches = 200;

static Mat ref;
static Mat tst;
static Mat diff;
static Mat rectifiedTst;
static std::vector<KeyPoint> keypointsRef;
static std::vector<KeyPoint> keypointsTst;
static std::vector<DMatch> bestMatches;

double get_seconds(std::chrono::high_resolution_clock::time_point t1,
                   std::chrono::high_resolution_clock::time_point t2)
{
    return (std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)).count();
}

void update_windows(int i = 0, void* ii = NULL)
{
    Mat ref_keypoints;
    drawKeypoints(ref, keypointsRef, ref_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow(refWinName, ref_keypoints);

    if (keypointsRef.size() > 0 and keypointsTst.size() > 0 and bestMatches.size() > 0) {
        Mat matches_img;
        drawMatches(ref, keypointsRef, tst, keypointsTst, bestMatches, matches_img);
        imshow(tstWinName, matches_img);
    }

    imshow(diffWinName, diff);
}

void detect(int, void*)
{
    Ptr<SURF> detector = SURF::create(minHessian, nOctaves, nOctLayers);

    // Find interest points in ref image and compute their descriptors
    auto refDetectionStart = std::chrono::high_resolution_clock::now();
    Mat descriptorsRef;
    detector->detectAndCompute(ref, Mat(), keypointsRef, descriptorsRef);
    auto refDetectionEnd = std::chrono::high_resolution_clock::now();

    // Find interest points in tst image and compute their descriptors
    auto tstDetectionStart = std::chrono::high_resolution_clock::now();
    Mat descriptorsTst;
    detector->detectAndCompute(tst, Mat(), keypointsTst, descriptorsTst);
    auto tstDetectionEnd = std::chrono::high_resolution_clock::now();

    // Find matching points and select n best of them
    auto matchStart = std::chrono::high_resolution_clock::now();
    std::vector<DMatch> matches;
    FlannBasedMatcher matcher;
    matcher.match(descriptorsRef, descriptorsTst, matches);
    auto matchComp = [] (const DMatch& a, const DMatch& b) -> bool {
        return a.distance < b.distance;
    };
    std::sort(matches.begin(), matches.end(), matchComp);
    bestMatches.clear();
    const int m = std::min(nMatches, (int)matches.size());
    for (int i = 0; i < m; i++) {
        bestMatches.push_back(matches[i]);
    }
    auto matchEnd = std::chrono::high_resolution_clock::now();

    // Calculate projective transformation from tst to ref
    auto homStart = std::chrono::high_resolution_clock::now();
    std::vector<Point2f> homSrc, homDst;
    for (int i = 0; i < bestMatches.size(); i++) {
        homDst.push_back(keypointsRef[bestMatches[i].queryIdx].pt);
        homSrc.push_back(keypointsTst[bestMatches[i].trainIdx].pt);
    }
    Mat homography = findHomography(homSrc, homDst, CV_RANSAC);
    auto homEnd = std::chrono::high_resolution_clock::now();

    if (homography.empty()) {
        std::cout << "Failed co compute homography matrix" << std::endl;
    }

    // Rectify test image
    warpPerspective(tst, rectifiedTst, homography, ref.size());

    // Calculate difference between ref and rectified tst
    Mat mask = (rectifiedTst != 0);
    Mat tmpDiff = Mat::zeros(ref.size(), CV_32F);
    Mat tmpFloat;
    ref.convertTo(tmpFloat, CV_32F);
    Mat tmpRect;
    rectifiedTst.convertTo(tmpRect, CV_32F);
    add(tmpFloat, -tmpRect, tmpDiff, mask, CV_32F);
    tmpDiff = abs(tmpDiff);

    Mat sq;
    multiply(tmpDiff, tmpDiff, sq);
    double mse = sqrt(sum(sq)[0]);

    tmpDiff.convertTo(diff, CV_8U);


    // Print info
    std::cout << "SURF feature detection. " << std::endl;
    std::cout << "    min hessian value: " << minHessian << std::endl;
    std::cout << "    number of octaves: " << nOctaves << std::endl;
    std::cout << "    layers per octave: " << nOctLayers << std::endl;
//    std::cout << "Iteration " << iteration << std::endl;
    std::cout << "Ref image keypoints: " << keypointsRef.size() << ", time: " <<
                 get_seconds(refDetectionStart, refDetectionEnd) << std::endl;
    std::cout << "Tst image keypoints: " << keypointsTst.size() << ", time: " <<
                 get_seconds(tstDetectionStart, tstDetectionEnd) << std::endl;
    std::cout << "Matching : FLANN-based, matches: " << matches.size() <<
                 ", time: " << get_seconds(matchStart, matchEnd) << std::endl;
    std::cout << "Homography time: " << get_seconds(homStart, homEnd) << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    update_windows();
}

void initWindows()
{
    namedWindow(refWinName, WINDOW_NORMAL);
    namedWindow(tstWinName, WINDOW_NORMAL);
    namedWindow(diffWinName, WINDOW_NORMAL);

    minHessian = 800;
    nOctaves = 4;
    nOctLayers = 2;
    nMatches = 200;

    createTrackbar(minHessianTrackbarName, refWinName, &minHessian, maxMinHessian, detect);
    createTrackbar(nOctavesTrackbarName, refWinName, &nOctaves, maxNOctaves, detect);
    createTrackbar(nOctLayersTrackbarName, refWinName, &nOctLayers, maxNOctLayers, detect);

    createTrackbar(nMatchesTrackbarName, tstWinName, &nMatches, maxNMatches, detect);

    imshow(refWinName, ref);
    imshow(tstWinName, tst);
}

void demoSURF(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
                             "{help h | | }{ @input-ref | example.jpg | }{ @input-tst | tst.jpg | }");
    if (parser.get<bool>("help")) {
        std::cout << "Help" << std::endl;
        return;
    }

    std::string refName = parser.get<std::string>("@input-ref");
    std::string tstName = parser.get<std::string>("@input-tst");
    std::cout << refName << std::endl;
    std::cout << tstName << std::endl;

    ref = imread(refName, CV_LOAD_IMAGE_GRAYSCALE);
    tst = imread(tstName, CV_LOAD_IMAGE_GRAYSCALE);
    if (ref.empty() or tst.empty()) {
        std::cout << "Failed to open image " << refName <<
                     " or " << tstName << std::endl;
        return;
    }

    initWindows();
    waitKey(0);
}
