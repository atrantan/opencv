// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>



class Detector
{
    enum Mode { Default, Daimler } m;
    cv::HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    std::string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    std::vector<cv::Rect> detect(cv::InputArray img)
    {
        // Run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        std::vector<cv::Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0.5, cv::Size(8,8), cv::Size(32,32), 1.05, 2, true);
        return found;
    }
    void adjustRect(cv::Rect & r) const
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};

static const std::string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use ot the HoG descriptor.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    // string file = parser.get<string>("video");
    std::string file = "vtest.avi";
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    cv::VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = cv::samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        std::cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << std::endl;
        return 2;
    }

    std::cout << "Press 'q' or <ESC> to quit." << std::endl;
    std::cout << "Press <space> to toggle between Default and Daimler detector" << std::endl;
    Detector detector;
    cv::Mat frame;
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Finished reading: empty frame" << std::endl;
            break;
        }
        int64 t = cv::getTickCount();
        std::vector<cv::Rect> found = detector.detect(frame);
        t = cv::getTickCount() - t;

        // show the window
        {
            std::ostringstream buf;
            buf << "Mode: " << detector.modeName() << " ||| "
                << "FPS: " << std::fixed << std::setprecision(1) << (cv::getTickFrequency() / (double)t);
            putText(frame, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        for (std::vector<cv::Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            cv::Rect &r = *i;
            detector.adjustRect(r);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
        }
        imshow("People detector", frame);

        // interact with user
        const char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            std::cout << "Exit requested" << std::endl;
            break;
        }
        else if (key == ' ')
        {
            detector.toggleMode();
        }
    }
    return 0;
}
