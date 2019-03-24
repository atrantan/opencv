//  Copyright (c) 2019 Antoine TRAN TAN

#ifndef MY_HEADER_H
#define MY_HEADER_H

#include <opencv2/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <string>

/* Global variables */
extern std::string face_cascade_name;
extern std::string eyes_cascade_name;
extern cv::CascadeClassifier face_cascade;
extern cv::CascadeClassifier eyes_cascade;
extern std::string window_name;
extern cv::RNG RNG;

/** Function Headers */
void detectAndDisplay( cv::Mat );

#endif
