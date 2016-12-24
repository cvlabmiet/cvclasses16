#include "stdafx.h"

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>
#include <iostream>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <ctime> 

using namespace cv;
using namespace std;

string fileName;

size_t numberOfPoints = 0;

Mat image;
Mat imageColor;
vector<Point> clickCoord; //не забыть очистить это всё и создать функцию для этого

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		circle(imageColor, Point(x, y), 1, Scalar(255, 0, 0), 3);
		imshow("original", imageColor);
		clickCoord.push_back(Point(x, y));
		numberOfPoints++;
	}

}

static void help()
{
	cout << "\nThis program demonstrates algorithm of Gabor filter for textures\n";
	cout << "Hot keys: \n"
		"\tESC or 'Q', 'q'- quit the program\n"
		"\t'R', 'r' - restore the original image\n"
		"\t'N', 'n' or SPACE - run gabor filter for texture segmentation algorithm\n"
		"\t\t(after running it, mark the areas of cluster centers on the image)\n";
		
}

cv::Mat showHist(Mat &gray)
{
	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Show the calculated histogram in command window
	double total;
	total = gray.rows * gray.cols;
	/*for (int h = 0; h < histSize; h++)
	{
	float binVal = hist.at<float>(h);
	cout << " " << binVal;
	}*/

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	//imshow(String("Result") + imageName, histImage);
	return histImage;

}
enum ConvolutionType
{
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,
	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,
	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};
void conv2(const Mat &img, const Mat& kernel, /*ConvolutionType type,*/ Mat& dest)
{
	Mat source = img;
	////if (CONVOLUTION_FULL == type)
	//{
	//	source = Mat();
	//	const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
	//	copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar(0));
	//}

	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;
	Mat temp;
	flip(kernel, temp, 1);
	Mat tempDest;
	filter2D(source, tempDest, img.depth(), temp, anchor, 0, borderMode);
	dest = tempDest(Rect(10, 10, tempDest.rows - 20, tempDest.cols - 20));
	//if (CONVOLUTION_VALID == type)
	//{
	//dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	//}
}
void  gabor2(Mat &image, Mat &imageGaborFilter, double gamma, double lambda, int b, double theta, double phi)
{
	double sigma = (1. / CV_PI) * sqrt(log(2) / 2) * (double(1 << b) + 1) / ((double)(1 << b) - 1) * lambda;
	int Sy = trunc(sigma * gamma);
	int ks = Sy * 2 + 1;
	Mat kernel(ks, ks, CV_32F);

	for (int x = -trunc(sigma); x <= trunc(sigma); x++) {//здесь изменила
		for (int y = -Sy; y <= Sy; y++) {//здесь ихменила
			double xp = (double)x * cos(theta) + (double)y * sin(theta);
			double yp = (double)y * cos(theta) - (double)x * sin(theta);
			kernel.at<float>(Sy + y, trunc(sigma) + x) = (float)exp(-0.5*(pow(xp, 2) +
				pow(gamma * yp, 2)) / pow(sigma, 2)) * cos(2 * CV_PI * xp / lambda + phi);
		}
	}

	conv2(image, kernel, /*ConvolutionType type,*/ imageGaborFilter);
}
cv::Mat gauss2(Mat &image, double sigma)
{
	int ks = trunc(sigma) * 2 + 1;
	int sigmaRound = trunc(sigma);
	Mat kernel(ks, ks, CV_32F);
	for (int x = -sigmaRound; x <= sigmaRound; x++) {
		for (int y = -sigmaRound; y <= sigmaRound; y++) {
			kernel.at<float>(sigmaRound + x, sigmaRound + y) = (float)exp(-0.5 * (pow(x / sigma, 2) + pow(y / sigma, 2)));
		}
	}

	Mat imageGaussFilter;
	conv2(image, kernel, /*ConvolutionType type,*/ imageGaussFilter);
	return imageGaussFilter;
}
cv::Mat eucdist2(Mat &X, Mat &Y)
{
	Mat U(Y.size(), CV_32F, Scalar(0));
	Mat V(X.size(), CV_32F, Scalar(0));
	Mat Xc, Yc;
	X.copyTo(Xc);
	Y.copyTo(Yc);
	for (int i = 0; i < Yc.rows; i++){
		U.at<float>(i) = !isnan(Yc.at<float>(i));
		if (!U.at<float>(i))
			Yc.at<float>(i) = 0;
	}

	for (int i = 0; i < Xc.rows; i++){
		V.at<float>(i) = !isnan(Xc.at<float>(i));
		if (!V.at<float>(i))
			Xc.at<float>(i) = 0;
	}
	Mat Yt = Yc.t();
	Mat Xt = Xc.t();
	Mat x2 = Xc.mul(Xc);
	Mat x2u = x2 * U.t();
	Mat y2 = Yt.mul(Yt);
	Mat vy2 = V * y2;
	Mat xyt = 2 * Xc * Yt;
	Mat d = x2u + vy2 - xyt;
	return d;
}
void kmeans_light(Mat &dataCluster, /*Mat &codebook,*/ Mat &data, Point sizeImage/*, size_t K*/)
{
	double stopIter = 0.05;
	Mat codebook;/* (K, 1, CV_32F);*/

	//Initial codebook
	if (numberOfPoints == 0){
		numberOfPoints = 5;
		Mat codebookLock(numberOfPoints, 1, CV_32F);
		srand(time(NULL));
		for (int i = 0; i < numberOfPoints; i++){
			int k = rand() % data.rows;
			codebookLock.at<float>(i) = data.at<float>(k);
		}
		codebookLock.copyTo(codebook);
	}
	else{
		Mat codebookLock(numberOfPoints, 1, CV_32F);
		for (int i = 0; i < numberOfPoints; i++){
			Point indx = clickCoord[i];
			codebookLock.at<float>(i) = data.at<float>(indx.x + sizeImage.x * indx.y);
		}
		codebookLock.copyTo(codebook);
	}
	
	float improvedRatio = INFINITY;
	float distortion = INFINITY;
	int iter = 0;

	float old_distortion;

	Mat dataNearClaserDist(data.rows, 1, CV_32F);
	//Mat dataCluster(data.rows, 1, CV_32F);


	while (true){
		//calculate euclidean distances between each sample and each codeword
		Mat d = eucdist2(data, codebook);

		//выбираем наименьшее число в каждой строке
		for (int i = 0; i < d.rows; i++){
			float min_val = INFINITY;
			for (int j = 0; j < numberOfPoints; j++){
				if (min_val > d.at<float>(i, j))
				{
					min_val = d.at<float>(i, j);
					dataCluster.at<int>(i) = j;
				}
			}
			dataNearClaserDist.at<float>(i) = min_val;
		}
		//distortion.If centroids are unchanged, distortion is also unchanged.
		//smaller distortion is better
		old_distortion = distortion;
		distortion = 0;
		for (int i = 0; i < dataNearClaserDist.rows; i++)
		{
			distortion += dataNearClaserDist.at<float>(i);
		}
		distortion /= dataNearClaserDist.rows;
		improvedRatio = 1 - (distortion / old_distortion);
		iter += 1;
		// If no more improved, break
		if (improvedRatio <= stopIter)
			break;
		// Renew codebook
		for (int i = 0; i < numberOfPoints; i++){
			std::vector<int> idx;
			int sizrvec = 0;
			for (int j = 0; j < dataNearClaserDist.rows; j++){
				if (dataCluster.at<int>(j) == i){
					idx.push_back(j);
					sizrvec++;
				}
			}
			if (idx.empty())
				continue;

			for (int k = 0; k < sizrvec; k++)
			{
				codebook.at<float>(i) += data.at<float>(idx[k]);
			}
			codebook.at<float>(i) /= idx.size();
		}
	}
	//return dataClaster;
}
int gabortexture(int argc, char** argv)
{
	help();
	while (true){
		char key;
		Mat hist;
		clock_t start, end;
		clock_t endSum = 0;

		string fileName;
		if (argc == 2) {
			fileName = argv[1];
		}
		image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
		if (image.empty())
		{
			cout << "Couldn'g open image " << fileName << ". Usage: gabortexture <image_name>\n";
			return 0;
		}
		cvtColor(image, imageColor, CV_GRAY2BGR);
		hist = showHist(image);
		imwrite("Historgamma image.jpg", hist);
		imshow("Historgamma image", hist);

		namedWindow("original", WINDOW_NORMAL);
		imshow("original", imageColor);
		setMouseCallback("original", CallBackFunc, NULL);
		key = (char)cv::waitKey();
		switch (key){
		case 27:
		case 'q':
		case 'Q':
			cout << "Exit from algorithm." << endl;
			return 0;

		case 'n':
		case 'N':
		case 32:
			cout << "Demonstration started || continued." << endl;
			break;

		case 'r':
		case 'R':
		default:
			std::cout << "Demonstration restored." << std::endl;
			numberOfPoints = 0;
			clickCoord.clear();
			continue;
		}

		double theta = 0.0;
		double gamma = 1.0;
		double lambda = 4.0904;
		int b = 1;
		double phi = 0;

		//Получаем фильтр габора
		Mat filterGabor;
		filterGabor.convertTo(filterGabor, CV_32F);
		image.convertTo(image, CV_32F);
		//Mat imageFilterGabor(image.rows, image.cols, CV_32F);
		Mat imageGaborFilter;

		start = clock();
		gabor2(image, imageGaborFilter, gamma, lambda, b, theta, phi);
		endSum += clock() - start;

		//Сглаживание изображения к которому применили фильтр Габора
		double minVal;
		double maxVal;

		minMaxLoc(imageGaborFilter, &minVal, NULL, NULL, NULL);
		add(imageGaborFilter, -minVal, imageGaborFilter);

		minMaxLoc(imageGaborFilter, NULL, &maxVal, NULL, NULL);
		divide(imageGaborFilter, maxVal, imageGaborFilter);

		minMaxLoc(imageGaborFilter, &minVal, &maxVal, NULL, NULL);

		float *p;
		for (int i = 0; i < imageGaborFilter.rows; ++i)
		{
			p = imageGaborFilter.ptr<float>(i);
			for (int j = 0; j < imageGaborFilter.cols; ++j)
			{
				double ans = tanh(p[j]);
				p[j] = ans;
			}
		}

		minVal = 0;
		maxVal = 0;
		minMaxLoc(imageGaborFilter, &minVal, NULL, NULL, NULL);
		add(imageGaborFilter, -minVal, imageGaborFilter);
		minMaxLoc(imageGaborFilter, NULL, &maxVal, NULL, NULL);
		divide(imageGaborFilter, maxVal, imageGaborFilter);

		minVal = 0;
		maxVal = 0;
		minMaxLoc(imageGaborFilter, &minVal, &maxVal, NULL, NULL);


		namedWindow("image Gabor", WINDOW_NORMAL);
		imshow("image Gabor", imageGaborFilter);
		Mat imGabFilter;
		imageGaborFilter.convertTo(imGabFilter, CV_8U, 255.0);
		imwrite(fileName + String("Image Gabor.jpg"), imGabFilter);

		hist = showHist(imGabFilter);
		imwrite("Historgamma image Gabor.jpg", hist);
		imshow("Historgamma image Gabor", hist);

		key = (char)cv::waitKey();
		switch (key){
		case 27:
		case 'q':
		case 'Q':
			cout << "Exit from algorithm." << endl;
			return 0;
		case 'n':
		case 'N':
		case 32:
			cout << "Demonstration started || continued." << endl;
			break;
		case 'r':
		case 'R':
		default:
			std::cout << "Demonstration restored." << std::endl;
			numberOfPoints = 0;
			clickCoord.clear();
			cv::destroyWindow("original");
			cv::destroyWindow("image Gabor");
			cv::destroyWindow("image blur");
			cv::destroyWindow("clustered image");
			cv::destroyWindow("Historgamma image Gabor Blur");
			cv::destroyWindow("Historgamma image Gabor");
			continue;
		}
		double sigma = (1.0 / CV_PI) * sqrt(log(2.0) / 2.0) * ((double)(1 << b) + 1) / ((double)(1 << b) - 1) * lambda;
		sigma = 3.0 * sigma;
		Mat imageGaborFilterBlur(imageGaborFilter.rows, imageGaborFilter.cols, CV_32F);

		start = clock();
		imageGaborFilterBlur = gauss2(imageGaborFilter, sigma);
		endSum += clock() - start;

		minVal = 0;
		maxVal = 0;
		minMaxLoc(imageGaborFilterBlur, &minVal, NULL, NULL, NULL);
		add(imageGaborFilterBlur, -minVal, imageGaborFilterBlur);
		minMaxLoc(imageGaborFilterBlur, NULL, &maxVal, NULL, NULL);
		divide(imageGaborFilterBlur, maxVal, imageGaborFilterBlur);

		namedWindow("image blur", WINDOW_NORMAL);
		imshow("image blur", imageGaborFilterBlur);
		Mat imGabFilterBlur;
		imageGaborFilterBlur.convertTo(imGabFilterBlur, CV_8U, 255.0);


		hist = showHist(imGabFilterBlur);
		imwrite("Historgamma image Gabor Blur.jpg", hist);
		imshow("Historgamma image Gabor Blur", hist);

		imwrite(fileName + String("Image Gabor Blur.jpg"), imGabFilterBlur);

		key = (char)cv::waitKey();
		switch (key){
		case 27:
		case 'q':
		case 'Q':
			cout << "Exit from algorithm." << endl;
			return 0;
		case 'n':
		case 'N':
		case 32:
			cout << "Demonstration started || continued." << endl;
			break;
		case 'r':
		case 'R':
		default:
			std::cout << "Demonstration restored." << std::endl;
			numberOfPoints = 0;
			clickCoord.clear();
			cv::destroyWindow("original");
			cv::destroyWindow("image Gabor");
			cv::destroyWindow("image blur");
			cv::destroyWindow("clustered image");
			cv::destroyWindow("Historgamma image Gabor Blur");
			cv::destroyWindow("Historgamma image Gabor");

			continue;
		}

		//Из imageGaborFilterBlur делаем одномерный вектор
		Mat data(imageGaborFilterBlur.rows * imageGaborFilterBlur.cols, 1, CV_32F);
		for (int i = 0; i < imageGaborFilterBlur.rows; i++){
			for (int j = 0; j < imageGaborFilterBlur.cols; j++){
				data.at<float>(i + imageGaborFilterBlur.rows * j) = imageGaborFilterBlur.at<float>(i, j);//всё верно
			}
		}



		Mat dataCluster(data.rows, 1, CV_32S);
		start = clock();
		kmeans_light(dataCluster, /*codebook,*/ data, Point(imageGaborFilterBlur.rows, imageGaborFilterBlur.cols)/*, K*/);


		//Из dataCluster делаем трёхканальное изображение размера image 
		Mat seg(imageGaborFilterBlur.size(), CV_8UC3);
		uchar colorTab[5][3] =
		{
			{ 0, 0, 255 },
			{ 0, 255, 0 },
			{ 255, 100, 100 },
			{ 255, 0, 255 },
			{ 0, 255, 255 }
		};
		int index;
		for (int i = 0; i < imageGaborFilterBlur.rows; i++){
			for (int j = 0; j < imageGaborFilterBlur.cols; j++){
				index = dataCluster.at<int>(i + imageGaborFilterBlur.rows * j);
				seg.at<Vec3b>(i, j)[0] = colorTab[index][0];//всё верно
				seg.at<Vec3b>(i, j)[1] = colorTab[index][1];//всё верно
				seg.at<Vec3b>(i, j)[2] = colorTab[index][2];//всё верно
			}
		}
		endSum += clock() - start;
		cout << "\nAlgorithm's execution time :" << (double)endSum / CLOCKS_PER_SEC << endl;
		namedWindow("clustered image", WINDOW_NORMAL);
		imshow("clustered image", seg);


		Mat segsave;
		seg.convertTo(segsave, CV_8U, 255.0);

		imwrite(fileName + String("Clustered image.jpg"), segsave);
		key = (char)cv::waitKey();
		switch (key){
		case 27:
		case 'q':
		case 'Q':
			cout << "Exit from algorithm." << endl;
			return 0;
		case 'n':
		case 'N':
			cout << "Demonstration started || continued." << endl;
		case 'r':
		case 'R':
		default:
			std::cout << "Demonstration restored." << std::endl;
			numberOfPoints = 0;
			clickCoord.clear();
			cv::destroyWindow("original");
			cv::destroyWindow("image Gabor");
			cv::destroyWindow("image blur");
			cv::destroyWindow("clustered image");
			cv::destroyWindow("Historgamma image Gabor Blur");
			cv::destroyWindow("Historgamma image Gabor");
			continue;
		}

		cv::destroyWindow("original");
		cv::destroyWindow("image Gabor");
		cv::destroyWindow("image blur");
		cv::destroyWindow("clustered image");
		cv::destroyWindow("Historgamma image Gabor Blur");
		cv::destroyWindow("Historgamma image Gabor");
	}


	return 0;
}

