#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

#include <iostream>
#include <ctype.h>

#define RECT_SIZE 40
#define STAY_TIME 15
#define MOVE_DIST .5

using namespace cv;
using namespace std;

Point2f point;
bool add_new_point = false;

static void onMouse( int event, int x, int y, int, void*)
{
	if( event == EVENT_LBUTTONDOWN )
	{
		point = Point2f((float)x, (float)y);
		add_new_point = true;
	}
}


void draw_rectangles(Mat &image, const vector<Point2f> &points,
                     const vector<size_t> &move_counter)
{
	std::vector<Rect2i> all_rectangles;
	std::vector<Rect2i> merged_rectangles;

	for (size_t i = 0; i < move_counter.size(); ++i)
		if (move_counter[i] > STAY_TIME)
			all_rectangles.push_back(Rect2i(Point(points[i].x - RECT_SIZE / 2,
			                                      points[i].y - RECT_SIZE / 2),
			                                Size(RECT_SIZE, RECT_SIZE)));

	std::vector<int> mask(all_rectangles.size(), 1);

	for (size_t i = 0; i < all_rectangles.size(); ++i) {

		if (!mask[i])
			continue;

		Rect2i curr_rectangle = all_rectangles[i];

		for (size_t j = i + 1; j < all_rectangles.size(); ++j) {
			if (mask[j]) {
				if ((curr_rectangle & all_rectangles[j]).area()) {
					curr_rectangle = curr_rectangle | all_rectangles[j];
					mask[j] = 0;
				}
			}
		}

		merged_rectangles.push_back(curr_rectangle);
	}

	for (auto &rect : merged_rectangles)
		rectangle(image, rect, {0, 0, 255});
}


int main( int argc, char** argv )
{
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	const int MAX_COUNT = 100;
	bool auto_init = true;

	VideoCapture capture;

	cv::CommandLineParser parser(argc, argv, "{@input_video||}"
	                                         "{@input_background||}"
	                                         "{@output_video| out.avi |}");

	string input_video = parser.get<string>("@input_video");
	string input_background = parser.get<string>("@input_background");
	string output_video = parser.get<string>("@output_video");

	if(input_video.empty()
	        || input_background.empty()
	        || output_video.empty()) {
		cerr << "Empty input parameters." << endl;
		return -1;
	}

	Mat background = imread(input_background, CV_LOAD_IMAGE_GRAYSCALE);

	if (background.empty()) {
		cerr << "Empty background" << endl;
		return -1;
	}

	VideoWriter frame_writer;
	frame_writer.open(output_video, CV_FOURCC('M', 'P', '4', '3'),
	                  30, background.size());

	if (!frame_writer.isOpened()) {
		cerr << "Frame writer is not opened." << endl;
		return -1;
	}

	// Load video
	capture.open(input_video);

	if(!capture.isOpened()) {
		cerr << "Capture is not open.";
		return -1;
	}

	// Create window
	namedWindow( "LK Demo", 1 );
	setMouseCallback( "LK Demo", onMouse, 0 );

	Mat gray, prevGray, image, frame;

	vector<Point2f> points[2];
	vector<size_t> move_counter;

	while (true)
	{
		capture >> frame;
		if( frame.empty() )
			break;

		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		// Itit points
		if(auto_init) {
			points[0].clear();
			points[1].clear();
			move_counter.clear();

			vector<Point2f> bg_points;

			// Find background points
			goodFeaturesToTrack(background, bg_points, MAX_COUNT, 0.01, 10, Mat(), 3, 1, 0.04);
			cornerSubPix(background, bg_points, subPixWinSize, Size(-1,-1), termcrit);

			vector<Point2f> frame_points;

			// Find point of input frame
			goodFeaturesToTrack(gray, frame_points, MAX_COUNT, 0.01, 10, Mat(), 3, 1, 0.04);
			cornerSubPix(gray, frame_points, subPixWinSize, Size(-1,-1), termcrit);

			for (auto &cp:frame_points) {
				bool add_flag = true;
				for (auto &bp : bg_points) {
					if (norm(cp - bp) < 3.) {
						add_flag = false;
						break;
					}
				}

				if (add_flag)
					points[1].push_back(cp);
			}

			add_new_point = false;
		}
		else if( !points[0].empty() )
		{
			vector<uchar> status;
			vector<float> err;
			std::vector<size_t> new_move_counter;

			if (move_counter.empty())
				move_counter.resize(points[0].size(), 0);

			if(prevGray.empty())
				gray.copyTo(prevGray);

			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
								 3, termcrit, 0, 0.001);
			size_t i, k;
			for( i = k = 0; i < points[1].size(); i++ ) {
				if(add_new_point) {
					if( norm(point - points[1][i]) <= 5 ) {
						add_new_point = false;
					}
				}

				if(!status[i])
					continue;

				if (norm(points[0][i] - points[1][i]) < MOVE_DIST)
					new_move_counter.push_back(move_counter[i] + 1);
				else
					new_move_counter.push_back(0);

				points[1][k++] = points[1][i];

				circle(image, points[1][i], 3, Scalar(0,255,0), -1, 8);
			}

			if (k == 0) {
				points[1].clear();
				points[0].clear();
			} else {
				points[1].resize(k);
			}

			std::swap(move_counter, new_move_counter);
		}

		if( add_new_point && points[1].size() < (size_t)MAX_COUNT )
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
			points[1].push_back(tmp[0]);
			add_new_point = false;
		}

		auto_init = false;

		// Show result
		draw_rectangles(image, points[1], move_counter);
		imshow("LK Demo", image);
		frame_writer.write(image);

		// Parse input from keyboard
		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch( c )
		{
		case 'r':
			auto_init = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			move_counter.clear();
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}

	destroyAllWindows();

	return 0;
}
