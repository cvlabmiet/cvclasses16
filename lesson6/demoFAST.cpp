#include "stdafx.h"
#include "BRIEF.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <bitset>
#include <fstream>
#include "FAST.h"
#include "helper.h"
#include <chrono>

//#pragma comment (lib, "opencv_world300d.lib")
using namespace std;
using namespace cv;

// Frame width and height of the capture
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;

int treshold = 30;
int treshold_test = 20;
int level = 5;

CVLAB::BRIEF brief;
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

Mat img1, img2;

void featureDetect(const Mat& image, Mat& output_image, vector<KeyPoint>& kpt, int treshold)
{
	FastDetector fd;

	fd.detect(image, treshold);
	fd.express(output_image);
	kpt = fd.getFeaturePoints();
}

void computeDescriptors(const Mat& image, vector<KeyPoint>& kpt, cv::Mat& output_descr)
{
	IplImage* image1;

	cv::Mat img11;
	img11 = image.clone();

	/* Convert to one channel image */
	cvtColor(img11, img11, CV_BGR2GRAY, 1);

	image1 = cvCreateImage(cvSize(img11.cols, img11.rows), 8, 1);

	IplImage ipltemp1 = img11;
	cvCopy(&ipltemp1, image1);

	IplImage *img1resized = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT),
		IPL_DEPTH_8U, 1);
	cvResize(image1, img1resized, CV_INTER_LINEAR);

	IplImage *img1resizedSmooth = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT),
		IPL_DEPTH_8U, 1);
	cvSmooth(img1resized, img1resizedSmooth, CV_GAUSSIAN, 3, 0, 1.0, 0.0);

	vector<bitset<256> > descriptors;

	brief.getBriefDescriptors(descriptors, kpt, img1resized);

	cv::Mat descr(kpt.size(), 256, CV_32F);

	for (size_t i = 0; i < descr.rows; ++i)
	{
		for (size_t j = 0; j < descr.cols; ++j)
		{
			descr.at<float>(i, j) = static_cast<float>((descriptors[i])[j]);
		}
	}

	output_descr = descr;
}

void projectView(cv::Mat& image_mark, vector<KeyPoint>& kpt1, cv::Mat& image_ref_mark, vector<KeyPoint>& kpt2, cv::Mat& descr, cv::Mat& ref_descr, std::vector<cv::DMatch> good_matches)
{

	cv::Mat img_matches;
	drawMatches(image_mark, kpt1, image_ref_mark, kpt2,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		vector<char>());

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		// Get the keypoints from the good matches
		obj.push_back(kpt1[good_matches[i].queryIdx].pt);
		scene.push_back(kpt2[good_matches[i].trainIdx].pt);
	}

	cv::Mat H;
	try {
		H = cv::findHomography(obj, scene, CV_RANSAC);
	}
	catch (...) {
		cout << "------------------------------------------------------------------------------------------- " << endl;
	};
	//std::cout << H;

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(image_mark.cols, 0);
	obj_corners[2] = cvPoint(image_mark.cols, image_mark.rows);
	obj_corners[3] = cvPoint(0, image_mark.rows);
	std::vector<cv::Point2f> scene_corners(4);

	try {
		perspectiveTransform(obj_corners, scene_corners, H);
	}
	catch (...) {
		cout << "------------------------------------------------------------------------------------------- " << endl;
	};

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + cv::Point2f(image_mark.cols, 0), scene_corners[1] + cv::Point2f(image_mark.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + cv::Point2f(image_mark.cols, 0), scene_corners[2] + cv::Point2f(image_mark.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + cv::Point2f(image_mark.cols, 0), scene_corners[3] + cv::Point2f(image_mark.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + cv::Point2f(image_mark.cols, 0), scene_corners[0] + cv::Point2f(image_mark.cols, 0), cv::Scalar(0, 255, 0), 4);
	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
}

void on_trackbar(int, void*)
{
	matcher->clear();
	system("cls");
	vector<KeyPoint> kpt1, kpt2;
	Mat desc1, desc2;
	
	Mat img11(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
	Mat img22(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);

	resize(img1, img11, img11.size(), 0, 0, CV_INTER_LINEAR);
	resize(img2, img22, img22.size(), 0, 0, CV_INTER_LINEAR);

	cv::Mat image_mark = img11.clone();
	cv::Mat image_ref_mark = img22.clone();

	cv::Mat image_base = img11.clone();
	cv::Mat image_ref_base = img22.clone();

	auto start = std::chrono::system_clock::now();
	featureDetect(img11, image_base, kpt1, treshold);

	std::chrono::duration<double> elapsed_seconds_ref = std::chrono::system_clock::now() - start;
	auto time_start_detect_test = std::chrono::system_clock::now();

	featureDetect(img22, image_ref_base, kpt2, treshold_test);

	std::chrono::duration<double> elapsed_seconds_test = std::chrono::system_clock::now() - time_start_detect_test;

	cv::imshow("Processed Reference window", image_base);

	cv::imshow("Processed TEST window", image_ref_base);

	cv::Mat descr, ref_descr;

	computeDescriptors(img11, kpt1, descr);
	computeDescriptors(img22, kpt2, ref_descr);

	//Matching

	std::vector<cv::DMatch> good_matches;

	std::vector< cv::DMatch > matches;

	if (descr.type() != CV_32F)
	{
		descr.convertTo(descr, CV_32F);
	}

	if (ref_descr.type() != CV_32F)
	{
		ref_descr.convertTo(ref_descr, CV_32F);
	}

	matcher->match(descr, ref_descr, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descr.rows; ++i)
	{
		double dist = matches[i].distance;
		if ((dist < min_dist) && (dist != 0)) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

	double level_true = (min_dist + (level / 10.0) * (max_dist - min_dist));
	for (int i = 0; i < descr.rows; ++i)
	{
		if ((matches[i].distance < level_true))
		{
			good_matches.push_back(matches[i]);
		}
	}

	projectView(image_mark, kpt1, image_ref_mark, kpt2, descr, ref_descr, good_matches);

	std::chrono::duration<double> all = std::chrono::system_clock::now() - start;

	cout << setw(40) << "*** PARAMETERS ***\n\n";

	cout << setw(35) << "** FAST **\n";
	cout << setw(30) << "Treshold for reference: " << treshold << endl;
	cout << setw(30) << "Keypoints reference size : " << kpt1.size() << endl;
	cout << setw(30) << "Ref features detected elapsed time: " << to_string(elapsed_seconds_ref.count()) << endl;

	cout << setw(30) << "Treshold for test: " << treshold_test << endl;
	cout << setw(30) << "Keypoints test size : " << kpt2.size() << endl;
	cout << setw(30) << "Test features detected elapsed time: " << to_string(elapsed_seconds_test.count()) << endl;

	cout << endl << setw(35) << "** BRIEF **\n";
	cout << setw(30) << "Descriptor reference size : " << descr.size() << endl;
	cout << setw(30) << "Descriptor test size : " << ref_descr.size() << endl;

	cout << endl << setw(40) << "** FLANN + RANSAC **\n";
	cout << setw(30) << "Level: " << level_true << endl;
	cout << setw(30) << "Number of matched pairs: " << good_matches.size() << endl;
	cout << setw(30) << "Number of rejected pairs: " << abs(matches.size() - good_matches.size()) << endl;
	cout << setw(30) << "Max dist : " << max_dist << endl;
	cout << setw(30) << "Min dist : " << min_dist << endl << endl;
	cout << setw(30) << "Time of iteration: " << to_string(all.count()) << endl;
}

void demoFAST(int argc, char **argv)
{
	cv::CommandLineParser parser(argc, argv, "{ help h					|					| }"
		"{ @reference_image		| images/ref.png| }"
		"{ @test_image			| images/test1.jpg| }");
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}

	std::string ref_im = parser.get<std::string>(0);
	std::string t_im = parser.get<std::string>(1);
	std::cout << ref_im <<" " <<  t_im <<std::endl;
	img1 = imread(ref_im, -1);
	img2 = imread(t_im, -1);

	if (img1.empty() || img2.empty()) 
	{
		cout << "Could not read one of the two images !" << endl;
		exit(-1);
	}


	cv::namedWindow("Processed Reference window", 1);
	cv::namedWindow("Processed TEST window", 1);
	cv::namedWindow("Good Matches & Object detection", 1);

	cv::createTrackbar("Treshold", "Processed Reference window", &treshold, 80, on_trackbar);
	cv::createTrackbar("Treshold Test", "Processed TEST window", &treshold_test, 80, on_trackbar);
	cv::createTrackbar("Level", "Good Matches & Object detection", &level, 10, on_trackbar);

	on_trackbar(treshold, 0);

	if((char)waitKey() == 27) exit(0);
}
