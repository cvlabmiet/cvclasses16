// lesson4.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>

#pragma comment(lib, "opencv_world300d.lib")

void demoSIFT(int, char*[]) { std::cout << "SIFT is under construction"; }; // TODO implement it in separate file
void demoSURF(int, char*[]) { std::cout << "SURF is under construction"; }; // TODO implement it in separate file
void demoFAST(int, char**) ; // FAST and BRIEF

int main(int argc, char* argv[])
{
    std::cout << "Select the detector: \n"
        << "1) SIFT detector demo\n"
        << "2) SURF detector demo\n"
        << "3) FAST detector demo\n";

    bool exit = false;
    do
    {
       int demoId = 0;
       std::cin >> demoId;
       switch (demoId)
       {
       case 1: demoSIFT(argc, argv); break;
       case 2: demoSURF(argc, argv); break;
       case 3: demoFAST(argc, argv); break;
       default:
          exit = true;
          break;
       }
    }
    while (!exit);

    system("pause");

    return 0;
}