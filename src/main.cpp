#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <vector>

#include "my_header.h"

namespace fs = std::filesystem;

/** Global variables */
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

int main()
{
    // Generate .csv file
    {
        // Output file
        std::ofstream ostrm("my_csv_file.csv", std::ios_base::out);

        int label = 0;

        fs::path path = fs::current_path() / "docs";

        for (auto &&p : fs::directory_iterator(path))
        {
            for (auto &&q : fs::directory_iterator(p))
            {
                std::string entry = q.path().string() + ";" + std::to_string(label) + "\n";
                std::replace(entry.begin(), entry.end(), '\\', '/');
                ostrm << entry;
            }
            label++;
        }
    }

    // Retrieve eyes coordinates
    {
        cv::String face_cascade_name = "haarcascades/haarcascade_frontalface_alt2.xml";
        cv::String eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";

        int label = 0;

        //-- 1. Load the cascades
        if (!face_cascade.load(face_cascade_name))
        {
            std::cout << "--(!)Error loading face cascade\n";
            return -1;
        };
        if (!eyes_cascade.load(eyes_cascade_name))
        {
            std::cout << "--(!)Error loading eyes cascade\n";
            return -1;
        };

        std::ifstream istrm("my_csv_file.csv", std::ios_base::in);
        std::ofstream ostrm("my_csv_file_modified.csv", std::ios_base::out);
        std::string line;

        while (std::getline(istrm, line))
        {
            std::size_t sep = line.find(';');
            std::string path(
                line.begin(), line.begin() + sep);

            cv::Mat frame = cv::imread(path);

            //-- 3. Apply the classifier to the frame
            auto eye_centers = detectAndDisplay(frame);

            if(!eye_centers.empty())
            {
                std::string entry;
                if(eye_centers[1].x < eye_centers[0].x)
                {
                    entry = line + " " 
                    + std::to_string(eye_centers[1].x) + " " 
                    + std::to_string(eye_centers[1].y) + " " 
                    + std::to_string(eye_centers[0].x) + " " 
                    + std::to_string(eye_centers[0].y) + "\n";
                }
                else
                {
                   entry = line + " " 
                    + std::to_string(eye_centers[0].x) + " " 
                    + std::to_string(eye_centers[0].y) + " " 
                    + std::to_string(eye_centers[1].x) + " " 
                    + std::to_string(eye_centers[1].y) + "\n";
                }
                ostrm << entry;
            }
        }
    }

    return 0;
}

/** @function detectAndDisplay */
std::vector<cv::Point> detectAndDisplay(cv::Mat frame)
{
    std::vector<cv::Point> result;

    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Mat faceROI = frame_gray(faces[i]);

        //-- In each face, detect eyes
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);

        for (size_t j = 0; j < eyes.size(); j++)
        {
            cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            cv::circle(frame, eye_center, radius, cv::Scalar(255, 0, 0), 4);
            result.push_back(eye_center);
        }
    }

    cv::putText(
        frame,
        "Taper 'y' si les 2 yeux sont bien detectes", 
        cv::Point(10,50),
        cv::FONT_HERSHEY_SIMPLEX,
        1, 
        cv::Scalar(209, 80, 0, 255),
        2);

    cv::putText(
        frame,
        "Taper 'n' dans le cas contraire", 
        cv::Point(10,90),
        cv::FONT_HERSHEY_SIMPLEX,
        1, 
        cv::Scalar(209, 80, 0, 255),
        2);

    //-- Show what you got
    cv::imshow("Capture - Face detection", frame);

    if (cv::waitKey(0) != 'y')
    {
        return std::vector<cv::Point>{};
    }
    else
    {
        return result;
    }
}