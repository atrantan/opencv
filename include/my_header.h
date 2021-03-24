//  Copyright (c) 2019 Antoine TRAN TAN

#ifndef MY_HEADER_H
#define MY_HEADER_H

#include "opencv2/core.hpp"
#include <string>
#include <vector>

/** Function Headers */
static void read_csv(
    const std::string &filename, 
    std::vector<cv::Mat> &images, 
    std::vector<int> &labels, 
    char separator = ';');

#endif
