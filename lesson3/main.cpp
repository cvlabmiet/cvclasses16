///@File: main.cpp
///@Brief: Contains implementation of entry point of the application
///@Author: Roman Golovanov
///@Date: 08 September 2015

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <map>
#include <functional>

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

const std::string c_origWindowName = "Original Picture";
const std::string c_edgeWindowName = "Edge Picture";
const std::string recPrecFileName  = "recprec.txt";

int g_threshold = 50;
int g_operatorId = 0;
cv::Mat edgePicture;
cv::Mat handEdgeImage;
std::ofstream recPrecFile;
std::pair<int, int> g_lastThOp = { -1, -1 };

///@brief Edge operator specification
struct EdgeOperator
{
   std::string Name;
   float fx[3*3]; // kernel for horizontal derivation
   float fy[3*3]; // kernel for vertical derivation
};

const std::string LoGName = "LoG";
const std::string CannyName = "Canny";
const std::string DoGName = "DoG";

///@brief array of edge operators
std::vector<EdgeOperator> g_operatorSpec = {
   EdgeOperator{ "Robert\'s",       { -1,  1, 0 , 0, 0, 0, 0, 0, 0 },
                                    { -1,  0, 0,  1, 0, 0, 0, 0, 0 } },
   EdgeOperator{ "Robert\'s Cross", { -1,  0, 0 , 0, 1, 0, 0, 0, 0 },
                                    {  0, -1, 0,  1, 0, 0, 0, 0, 0 } },
   EdgeOperator{ "Prewitt",         { -1,  0,  1, -1, 0, 1, -1, 0, 1 },
                                    { -1, -1, -1,  0, 0, 0,  1, 1, 1 } },
   EdgeOperator{ "Sobel",           { -1,  0,  1, -2, 0, 2, -1, 0, 1 },
                                    { -1, -2, -1,  0, 0, 0,  1, 2, 1 } },
   EdgeOperator{ "Scharr",          { -3,   0,  3, -10, 0, 10, -3,  0, 3},
                                    { -3, -10, -3,   0, 0,  0,  3, 10, 3} },
   EdgeOperator{ LoGName,           { 1,  1, 1 , 1, -8, 1, 1, 1, 1 },
                                    { 0,  0, 0,  0,  0, 0, 0, 0, 0 } },
   EdgeOperator{ DoGName,           { 0.01130886, 0.07798381, 0.01130886,
                                      0.07798381, -0.3571707, 0.07798381,
                                      0.01130886, 0.07798381, 0.01130886},
                                    { 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
   EdgeOperator{ CannyName },
};

static void calcRecPrec(cv::Mat img, double* rec, double* prec)
{
//    cv::dilate(img, img, cv::Mat());
    size_t false_negs = 0;
    size_t true_negs = 0;
    size_t false_pos = 0;
    size_t true_pos = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<float>(i, j) > 0) {
                if (handEdgeImage.at<u_char>(i, j) > 0) {
                    true_pos++;
                }
                else {
                    false_pos++;
                }
            }
            else {
                if (handEdgeImage.at<u_char>(i, j) > 0) {
                    false_negs++;
                }
                else {
                    true_negs++;
                }
            }
        }
    }
    *rec = true_pos / ((double) true_pos + false_negs);
    *prec = true_pos / ((double) true_pos + false_pos);
}

static void onMouse(int event, int x, int y, int flags, void*)
{
    if (event != CV_EVENT_MBUTTONUP)
    {
        return;
    }

    auto& oper = g_operatorSpec[g_operatorId];
    const int bufSize = 100;
    char chint[bufSize];
    sprintf(chint, "%s_%d.bmp", oper.Name.data(), g_threshold);
    cv::imwrite(chint, edgePicture);


    double rec, prec;
    calcRecPrec(edgePicture, &rec, &prec);
    if (isnan(prec)) {
        prec = 1;
    }
    recPrecFile << chint << "  " << rec << "  " << prec << std::endl;
    std::cout << "Saved an image: " << chint << std::endl;
}

/////////////////////////////////////////////////////////////////////
void apply1stDerivativeAlgo(cv::Mat& i_img, EdgeOperator& i_oper, cv::Mat& o_edges)
{
   cv::Mat imgX;
   cv::filter2D(i_img, imgX, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fx));

   cv::Mat imgY;
   cv::filter2D(i_img, imgY, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fy));

   cv::Mat imgMag;
   cv::magnitude(imgX, imgY, imgMag);

   double maxVal = 0;
   cv::minMaxLoc(imgMag, nullptr, &maxVal);

   cv::threshold(imgMag, o_edges, 0.01 * g_threshold * maxVal, 1, CV_THRESH_BINARY);
}

/////////////////////////////////////////////////////////////////////
void apply2ndDerivativeAlgo(cv::Mat& i_img, EdgeOperator& i_oper, cv::Mat& o_edges)
{
   cv::Mat log;
   cv::filter2D(i_img, log, CV_32F, cv::Mat(3, 3, CV_32F, i_oper.fx));

   double minVal = 0;
   double maxVal = 0;
   cv::minMaxLoc(log, &minVal, &maxVal);
   const double th = 0.01 * g_threshold * (std::max)(maxVal, std::abs(minVal));

   o_edges = cv::Mat::zeros(log.size(), log.type());

   for (int x = 1; x < log.cols - 1; ++x)
   {
      for (int y = 1; y < log.rows - 1; ++y)
      {
         const cv::Mat nghb = log.colRange({ x - 1, x + 2 }).rowRange({ y - 1, y + 2 });

         if (nghb.at<float>(0, 0) * nghb.at<float>(2, 2) < 0 && std::abs(nghb.at<float>(0, 0) - nghb.at<float>(2, 2)) > th ||
            nghb.at<float>(1, 0) * nghb.at<float>(1, 2) < 0 && std::abs(nghb.at<float>(1, 0) - nghb.at<float>(1, 2)) > th ||
            nghb.at<float>(2, 0) * nghb.at<float>(0, 2) < 0 && std::abs(nghb.at<float>(2, 0) - nghb.at<float>(0, 2)) > th ||
            nghb.at<float>(2, 1) * nghb.at<float>(0, 1) < 0 && std::abs(nghb.at<float>(2, 1) - nghb.at<float>(0, 1)) > th)
         {
            o_edges.at<float>(y, x) = 1.0;
         }
      }
   }
}

///@brief provides frame processing
void processFrame(cv::Mat& i_image)
{
   if (g_lastThOp == std::make_pair(g_threshold, g_operatorId))
   {
      return; // nothing changed
   }

   g_lastThOp = { g_threshold, g_operatorId };

   auto& oper = g_operatorSpec[g_operatorId];

   cv::imshow(c_origWindowName, i_image);
   
   cv::Mat gray;
   cv::cvtColor(i_image, gray, CV_RGB2GRAY);

   cv::Mat edges;
   if (oper.Name == CannyName)
   {
      cv::Canny(gray, edges, g_threshold, g_threshold / 2);
   }
   else if(oper.Name == LoGName)
   {
      apply2ndDerivativeAlgo(gray, oper, edges);
   }
   else
   {
      apply1stDerivativeAlgo(gray, oper, edges);
   }
   edges.copyTo(edgePicture);
   edgePicture = edgePicture * 255;

   const int bufSize = 100;
   char chint[bufSize];
   sprintf(chint, "%s Operator, %d", oper.Name.data(), g_threshold);
   std::string shint{ chint };
   const auto txtSize = cv::getTextSize(shint, CV_FONT_HERSHEY_PLAIN, 1.0, 1, nullptr);
   cv::putText(edges, shint, { 10, 10 + txtSize.height }, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);

   cv::imshow(c_edgeWindowName, edges);
}

///@brief provides video capturing from the camera and frame processing
void processVideo()
{
	cv::VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::cerr << "Can not open the camera !" << std::endl;
		return;
	}

	while (cv::waitKey(1) < 0)
	{
		cv::Mat frame;
		capture.read(frame);
		processFrame(frame);
	}
}

///@brief provides image file processing
bool processImage(const std::string& i_name)
{

   cv::Mat img0 = cv::imread(i_name, 1);
   if (img0.empty())
   {
      std::cerr << "Couldn'g open image " << i_name << ". Usage: lesson3 <image_name>\n";
      return false;
   }

   while (cv::waitKey(1) < 0)
   {
      cv::Mat frame;
      processFrame(img0);
   }

   return true;
}

void trackBarCallBack(int, void*)
{
   printf("Operator: %s; Threshold: %d\n", g_operatorSpec[g_operatorId].Name.data(), g_threshold);
}

std::string type2str(int type) {
  std::string r;

  u_char depth = type & CV_MAT_DEPTH_MASK;
  u_char chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

///@brief Entry point
int main(int argc, char** argv)
{
   cv::CommandLineParser parser(argc, argv, "{ help h |  | }"
                                            "{ @input |  | }"
                                            "{ @handEdge |  | }");
   if (parser.has("help"))
   {
      parser.printMessage();
      return 0;
   }
   const auto& fileName = parser.get<std::string>("@input");
   const auto& handFileName = parser.get<std::string>("@handEdge");
   handEdgeImage = cv::imread(handFileName, CV_LOAD_IMAGE_GRAYSCALE);
   if (handEdgeImage.empty()) {
       std::cout << "Failed to open hand-drawn image" << std::endl;
   }

   /// Create window with original image
   cv::namedWindow(c_origWindowName, CV_WINDOW_NORMAL);
   cv::resizeWindow(c_origWindowName, 640, 480);

   /// Create DEMO window
   cv::namedWindow(c_edgeWindowName, CV_WINDOW_NORMAL);
   cv::resizeWindow(c_edgeWindowName, 640, 480);

   cv::createTrackbar("Threshold", c_edgeWindowName, &g_threshold, 100, trackBarCallBack);
   cv::createTrackbar("Operator",  c_edgeWindowName, &g_operatorId, g_operatorSpec.size() - 1, trackBarCallBack);
   cv::setMouseCallback(c_edgeWindowName, onMouse, 0);

   recPrecFile.open(recPrecFileName);

   std::cout << type2str(handEdgeImage.type()) << std::endl;

   if (!processImage(fileName))
   {
      processVideo();
   }

   recPrecFile.close();

	return 0;
}
