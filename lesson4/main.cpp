///@File: main.cpp
///@Brief: Contains interface of console application for testing background
///        subtraction algorithms
///@Author: Vitaliy Baldeev
///@Date: 01 October 2015

#include <iostream>
#include <memory>
#include <list>

#include "SegmentMotionDiff.h"
#include "SegmentMotionBU.h"
#include "SegmentMotionGMM.h"
#include "SegmentMotion1G.h"
#include "SegmentMotionMedian.h"

void ViBeDemo();

int main()
{
    std::cout << "Select the algorithm: \n"
        << "Diff  - Basic difference \n"
        << "BU    - Basic difference with background updating \n"
        << "GMM   - Gaussian mixture model algorithm \n"
        << "MM    - MinMax algoruthm \n"
        << "1G    - One Gaussian \n"
        << "VB    - ViBe algorithm \n"
		<< "Med   - Median Filtering algorithm\n";
    std::string algorithmName;
	std::cin >> algorithmName;
    std::unique_ptr<SegmentMotionBase> ptr(SegmentMotionBase::CreateAlgorithm(algorithmName));
    if (ptr)
    {
		std::cout << "enter video source (0 = camera)" << std::endl;
		std::string video_source;
		std::cin >> video_source;
		if (video_source == "0")
			ptr->Run();
		else
			ptr->Run(video_source);
    }
	else
	{
		std::cout << "Run ViBe by default" << std::endl;
		ViBeDemo();
	}

    return 0;
}
